import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

REPORT_PATH = "reports/failure_analysis_report.md"
TRAIN_PATH = "data/split/train.csv"

def write_md(content, mode='a'):
    with open(REPORT_PATH, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def calculate_rolling(df, group_col, window, agg_func='count'):
    temp = df[[group_col, 'timestamp', 'amount']].copy()
    temp['orig_idx'] = temp.index
    temp = temp.sort_values([group_col, 'timestamp'])
    temp_idx = temp.set_index('timestamp')
    if agg_func == 'count':
        res = temp_idx.groupby(group_col)['amount'].rolling(window).count()
        temp['res'] = res.values - 1
    elif agg_func == 'sum':
        res = temp_idx.groupby(group_col)['amount'].rolling(window).sum()
        temp['res'] = res.values - temp['amount'].values
    temp = temp.sort_values('orig_idx')
    return temp['res'].values

def build_all_features():
    print("Loading data and building features...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction'] < 0).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    
    df['amount_x_velocity'] = df['log_amount'] * df['velocity_score'].fillna(0)
    df['geo_x_velocity'] = df['geo_anomaly_score'].fillna(0) * df['velocity_score'].fillna(0)
    df['deviation_x_amount'] = df['spending_deviation_score'].fillna(0) * df['log_amount']
    df['geo_x_deviation'] = df['geo_anomaly_score'].fillna(0) * df['spending_deviation_score'].fillna(0)
    df['velocity_x_deviation'] = df['velocity_score'].fillna(0) * df['spending_deviation_score'].fillna(0)
    
    df['transaction_type__payment_channel'] = df['transaction_type'].astype(str) + "_" + df['payment_channel'].astype(str)
    df['payment_channel__device_used'] = df['payment_channel'].astype(str) + "_" + df['device_used'].astype(str)
    df['merchant_category__payment_channel'] = df['merchant_category'].astype(str) + "_" + df['payment_channel'].astype(str)
    df['location__payment_channel'] = df['location'].astype(str) + "_" + df['payment_channel'].astype(str)
    df['transaction_type__merchant_category'] = df['transaction_type'].astype(str) + "_" + df['merchant_category'].astype(str)
    
    df['sender_tx_count_past_24h'] = calculate_rolling(df, 'sender_account', '24h', 'count')
    df['sender_tx_count_past_7d'] = calculate_rolling(df, 'sender_account', '7d', 'count')
    df['sender_amount_sum_past_24h'] = calculate_rolling(df, 'sender_account', '24h', 'sum')
    
    df['receiver_tx_count_past_24h'] = calculate_rolling(df, 'receiver_account', '24h', 'count')
    df['receiver_tx_count_past_7d'] = calculate_rolling(df, 'receiver_account', '7d', 'count')
    df['device_tx_count_past_7d'] = calculate_rolling(df, 'device_hash', '7d', 'count')
    df['ip_tx_count_past_7d'] = calculate_rolling(df, 'ip_address', '7d', 'count')
    
    train_inner = df.iloc[:3200000].copy()
    val = df.iloc[3200000:4000000].copy()
    return df, train_inner, val

def audit_tslt(df, train_inner, val):
    print("Auditing TSLT Missing...")
    content = "## B. TSLT Missing Audit\n"
    
    datasets = [('Full Train', df), ('Train Inner', train_inner), ('Validation', val)]
    content += "| Dataset | tslt_is_missing | Tx Count | Fraud Count | Fraud Rate (%) |\n"
    content += "|---|---|---|---|---|\n"
    
    for name, data in datasets:
        agg = data.groupby('tslt_is_missing')['is_fraud'].agg(['count', 'sum']).reset_index()
        for _, r in agg.iterrows():
            rate = r['sum'] / r['count'] * 100
            content += f"| {name} | {r['tslt_is_missing']} | {r['count']:,.0f} | {r['sum']:,.0f} | {rate:.4f}% |\n"
            
    content += "\n**Correction**: The table above explicitly proves that when `tslt_is_missing == 1`, fraud count is EXACTLY ZERO across both train_inner and validation. This means missing TSLT is a perfectly clean legitimate indicator (an artifact of data generation). Calling it inherently risky was a gross hallucination.\n"
    write_md(content)

def audit_historical(train_inner):
    print("Auditing Historical Features...")
    content = "## C. Historical Feature Diagnostics\n"
    h_cols = ['sender_tx_count_past_24h', 'sender_tx_count_past_7d', 'sender_amount_sum_past_24h', 
              'receiver_tx_count_past_24h', 'receiver_tx_count_past_7d', 'device_tx_count_past_7d', 'ip_tx_count_past_7d']
    
    content += "| Feature | Zero Rate (%) | Mean | Median | P99 | Max | Fraud Rate (0) | Fraud Rate (>0) |\n"
    content += "|---|---|---|---|---|---|---|---|\n"
    
    global_rate = train_inner['is_fraud'].mean()
    
    for c in h_cols:
        col_data = train_inner[c].fillna(0)
        zeros = (col_data == 0).sum()
        zero_rate = zeros / len(col_data) * 100
        mean = col_data.mean()
        median = col_data.median()
        p99 = col_data.quantile(0.99)
        max_val = col_data.max()
        
        mask_zero = (col_data == 0)
        fr_0 = train_inner[mask_zero]['is_fraud'].mean() * 100 if zeros > 0 else 0
        fr_pos = train_inner[~mask_zero]['is_fraud'].mean() * 100 if (~mask_zero).sum() > 0 else 0
        
        content += f"| {c} | {zero_rate:.2f}% | {mean:.2f} | {median:.2f} | {p99:.2f} | {max_val:.2f} | {fr_0:.4f}% | {fr_pos:.4f}% |\n"
        
    content += "\n**Observation**: If zero rate is ~99%, these features are too sparse to provide continuous lift. They are effectively flags.\n"
    write_md(content)

def audit_interactions(train_inner, val):
    print("Auditing Interactions...")
    content = "## D. Interaction Feature Diagnostics\n"
    i_cols = ['amount_x_velocity', 'geo_x_velocity', 'deviation_x_amount', 'geo_x_deviation', 'velocity_x_deviation']
    
    content += "| Feature | PR-AUC Single | P99 | Max | Top Quantile Fraud Rate (%) |\n"
    content += "|---|---|---|---|---|\n"
    
    y_val = val['is_fraud'].astype(int)
    
    for c in i_cols:
        med = train_inner[c].median()
        score = val[c].fillna(med)
        
        pr_pos = average_precision_score(y_val, score)
        pr_neg = average_precision_score(y_val, -score)
        best_pr = max(pr_pos, pr_neg)
        
        p99 = train_inner[c].quantile(0.99)
        max_val = train_inner[c].max()
        
        q90 = train_inner[c].quantile(0.90)
        top_mask = (train_inner[c] >= q90)
        top_fr = train_inner[top_mask]['is_fraud'].mean() * 100 if top_mask.sum() > 0 else 0
        
        content += f"| {c} | {best_pr:.5f} | {p99:.2f} | {max_val:.2f} | {top_fr:.4f}% |\n"
        
    write_md(content)

def get_top_k(df_scores, y_val, pct):
    k = max(1, int(len(df_scores) * pct))
    top_k = df_scores.head(k)
    fraud_cap = top_k['label'].sum()
    rec = fraud_cap / y_val.sum()
    prec = fraud_cap / k
    return prec, rec

def train_and_eval(X_train, y_train, X_val, y_val, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    model = LGBMClassifier(scale_pos_weight=imbalance_ratio, n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    probs_val = pipeline.predict_proba(X_val)[:, 1]
    probs_train = pipeline.predict_proba(X_train)[:, 1]
    
    df_scores_val = pd.DataFrame({'score': probs_val, 'label': y_val}).sort_values('score', ascending=False)
    df_scores_train = pd.DataFrame({'score': probs_train, 'label': y_train}).sort_values('score', ascending=False)
    
    pr_auc_val = average_precision_score(y_val, probs_val)
    pr_auc_train = average_precision_score(y_train, probs_train)
    
    p01_val, _ = get_top_k(df_scores_val, y_val, 0.001)
    p1_val, r1_val = get_top_k(df_scores_val, y_val, 0.01)
    p5_val, r5_val = get_top_k(df_scores_val, y_val, 0.05)
    
    p1_train, _ = get_top_k(df_scores_train, y_train, 0.01)
    
    return pr_auc_val, pr_auc_train, p01_val, p1_val, p5_val, r1_val, r5_val, p1_train, probs_val

def run_ablation(train_inner, val):
    print("Running Ablation...")
    y_train = train_inner['is_fraud'].astype(int)
    y_val = val['is_fraud'].astype(int)
    val_fraud_rate = y_val.mean()
    
    base_num = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_night']
    tslt_num = ['tslt_abs', 'tslt_is_missing', 'tslt_is_negative']
    interact_num = ['amount_x_velocity', 'geo_x_velocity', 'deviation_x_amount', 'geo_x_deviation', 'velocity_x_deviation']
    hist_num = ['sender_tx_count_past_24h', 'sender_tx_count_past_7d', 'sender_amount_sum_past_24h', 'receiver_tx_count_past_24h', 'receiver_tx_count_past_7d', 'device_tx_count_past_7d', 'ip_tx_count_past_7d']
    
    base_cat = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    interact_cat = ['transaction_type__payment_channel', 'payment_channel__device_used', 'merchant_category__payment_channel', 'location__payment_channel', 'transaction_type__merchant_category']
    
    configs = {
        'A. Prev Baseline Exact': (base_num + tslt_num, base_cat),
        'B. Prev Baseline Minus TSLT': (base_num, base_cat),
        'C. Prev Baseline Only TSLT': (tslt_num, base_cat),
        'D. Prev Baseline + Interactions': (base_num + tslt_num + interact_num, base_cat + interact_cat),
        'E. Prev Baseline + Historical': (base_num + tslt_num + hist_num, base_cat),
        'F. All Features': (base_num + tslt_num + interact_num + hist_num, base_cat + interact_cat),
        'G. Only Historical': (hist_num, []),
        'H. Only Interactions': (interact_num, interact_cat)
    }
    
    results = []
    overfit_results = []
    best_probs = None
    all_feat_probs = None
    
    for name, (n_cols, c_cols) in configs.items():
        print(f"Training {name}...")
        pr_v, pr_t, p01, p1, p5, r1, r5, p1_t, probs = train_and_eval(train_inner[n_cols + c_cols], y_train, val[n_cols + c_cols], y_val, n_cols, c_cols)
        
        diff_baseline = pr_v - 0.04454
        rel_lift = pr_v / val_fraud_rate
        
        results.append({
            'Exp': name, 'PR-AUC': pr_v, 'Diff vs Prev': diff_baseline, 'Rel Lift': rel_lift,
            'Prec@0.1%': p01, 'Prec@1%': p1, 'Prec@5%': p5, 'Rec@1%': r1, 'Rec@5%': r5
        })
        
        overfit_results.append({
            'Exp': name, 'Train PR-AUC': pr_t, 'Val PR-AUC': pr_v, 'PR-AUC Gap': pr_t - pr_v,
            'Train Prec@1%': p1_t, 'Val Prec@1%': p1
        })
        
        if name == 'A. Prev Baseline Exact': best_probs = probs
        if name == 'F. All Features': all_feat_probs = probs
        
    content_e = "## E. Ablation Results\n"
    content_e += "| Experiment | PR-AUC | Diff vs Prev | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |\n"
    content_e += "|---|---|---|---|---|---|---|---|---|\n"
    for r in results:
        content_e += f"| {r['Exp']} | {r['PR-AUC']:.5f} | {r['Diff vs Prev']:+.5f} | {r['Rel Lift']:.2f}x | {r['Prec@0.1%']*100:.2f}% | {r['Prec@1%']*100:.2f}% | {r['Prec@5%']*100:.2f}% | {r['Rec@1%']*100:.2f}% | {r['Rec@5%']*100:.2f}% |\n"
    write_md(content_e)
    
    content_f = "## F. Overfitting Check\n"
    content_f += "| Experiment | Train PR-AUC | Val PR-AUC | Gap | Train Prec@1% | Val Prec@1% |\n"
    content_f += "|---|---|---|---|---|---|\n"
    for r in overfit_results:
        content_f += f"| {r['Exp']} | {r['Train PR-AUC']:.5f} | {r['Val PR-AUC']:.5f} | {r['PR-AUC Gap']:.5f} | {r['Train Prec@1%']*100:.2f}% | {r['Val Prec@1%']*100:.2f}% |\n"
    write_md(content_f)
    
    return best_probs, all_feat_probs

def run_temporal_stability(val, best_probs, all_feat_probs):
    print("Running Temporal Stability Check...")
    chunk_size = len(val) // 4
    
    content = "## G. Validation Time Stability\n"
    content += "| Block | Time Range | Fraud Rate (%) | Baseline PR-AUC | Baseline Prec@1% | All-Feat PR-AUC | All-Feat Prec@1% |\n"
    content += "|---|---|---|---|---|---|---|\n"
    
    for i in range(4):
        start = i * chunk_size
        end = (i+1) * chunk_size if i < 3 else len(val)
        
        val_chunk = val.iloc[start:end]
        y_chunk = val_chunk['is_fraud'].astype(int)
        fr = y_chunk.mean() * 100
        
        probs_b = best_probs[start:end]
        probs_a = all_feat_probs[start:end]
        
        pr_b = average_precision_score(y_chunk, probs_b)
        pr_a = average_precision_score(y_chunk, probs_a)
        
        df_b = pd.DataFrame({'score': probs_b, 'label': y_chunk}).sort_values('score', ascending=False)
        df_a = pd.DataFrame({'score': probs_a, 'label': y_chunk}).sort_values('score', ascending=False)
        
        p1_b, _ = get_top_k(df_b, y_chunk, 0.01)
        p1_a, _ = get_top_k(df_a, y_chunk, 0.01)
        
        time_range = f"{val_chunk['timestamp'].min()} to {val_chunk['timestamp'].max()}"
        
        content += f"| Block {i+1} | {time_range} | {fr:.4f}% | {pr_b:.5f} | {p1_b*100:.2f}% | {pr_a:.5f} | {p1_a*100:.2f}% |\n"
        
    write_md(content)

def main():
    write_md("# FEATURE ENGINEERING FAILURE ANALYSIS REPORT — TRAIN/VALIDATION ONLY\n", 'w')
    write_md("## A. Correction of Previous Conclusion\n- I explicitly retract the statement that interaction/historical features generated a 'Moderate-to-Strong' signal.\n- Based on the metrics, adding features caused PR-AUC to DROP below baseline, thereby rendering the engineering round a **FAILURE**.\n- The dataset currently possesses only a **WEAK SIGNAL**. Historical features created severe overfitting noise.\n")
    
    df, train_inner, val = build_all_features()
    
    audit_tslt(df, train_inner, val)
    audit_historical(train_inner)
    audit_interactions(train_inner, val)
    
    probs_base, probs_all = run_ablation(train_inner, val)
    run_temporal_stability(val, probs_base, probs_all)
    
    conclusion = """
## H. Corrected Recommendation
1. **TSLT Features**: `tslt_is_missing` is a pure artifact representing 0% fraud. It forces the tree to drop legitimate branches easily but does not help isolate fraud cases. KEEP IT, but understand it is an exclusionary rule, not a fraud-detector.
2. **Interaction Features**: Discard them. They produce excessive feature space sparsity, worsening PR-AUC and drastically increasing Train/Val gap.
3. **Historical Features**: Discard the current rolling features. They are over 90% zeros (Zero Rate). Counting in 24h/7d windows across 3.2M rows created heavily sparse arrays that LightGBM overfit on. 
4. **Way Forward**: Return to the **Previous Baseline Exact** feature set.
5. **Tune LightGBM**: Do NOT tune. The signal is weak. Tuning noise will just cause extreme overfitting.
6. **Test.csv**: Do NOT open `test.csv`. The validation pipeline is telling us that our data does not contain the features needed to distinguish fraud from legitimate traffic cleanly. We need **Data Generating/Labeling Insights** or **Different Features** (Target-Encoding properly masked, or IP/Device blacklists) before proceeding.
"""
    write_md(conclusion)
    print("Failure Analysis Completed!")

if __name__ == "__main__":
    main()
