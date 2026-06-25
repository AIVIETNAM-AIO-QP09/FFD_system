import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

REPORT_PATH = "reports/target_encoding_report.md"
TRAIN_PATH = "data/split/train.csv"

def write_md(content, mode='a'):
    with open(REPORT_PATH, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def get_top_k(df_scores, y_val, pct):
    k = max(1, int(len(df_scores) * pct))
    top_k = df_scores.head(k)
    fraud_cap = top_k['label'].sum()
    rec = fraud_cap / y_val.sum() if y_val.sum() > 0 else 0
    prec = fraud_cap / k
    return prec, rec

def calculate_target_encoding(df, group_col, window, alpha, global_rate):
    temp = df[[group_col, 'timestamp', 'is_fraud']].copy()
    temp['orig_idx'] = temp.index
    temp = temp.sort_values([group_col, 'timestamp'])
    temp_idx = temp.set_index('timestamp')
    
    # Rolling counts and sums
    count_series = temp_idx.groupby(group_col)['is_fraud'].rolling(window).count()
    sum_series = temp_idx.groupby(group_col)['is_fraud'].rolling(window).sum()
    
    # Subtract current transaction
    temp['tx_count'] = count_series.values - 1
    temp['fraud_count'] = sum_series.values - temp['is_fraud'].values
    
    # Smoothed rate
    temp['smoothed_rate'] = (temp['fraud_count'] + alpha * global_rate) / (temp['tx_count'] + alpha)
    
    temp = temp.sort_values('orig_idx')
    return temp['fraud_count'].values, temp['smoothed_rate'].values

def prepare_data():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Base features
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction'] < 0).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    
    # Global Rate from first 3.2M only to avoid leakage into target encoding
    train_inner_temp = df.iloc[:3200000]
    global_rate = train_inner_temp['is_fraud'].mean()
    alpha = 10
    
    print("Calculating Target Encoding Features...")
    # Sender
    df['sender_fraud_count_past_7d'], df['sender_fraud_rate_past_7d'] = calculate_target_encoding(df, 'sender_account', '7d', alpha, global_rate)
    df['sender_fraud_count_past_30d'], df['sender_fraud_rate_past_30d'] = calculate_target_encoding(df, 'sender_account', '30d', alpha, global_rate)
    
    # Receiver
    df['receiver_fraud_count_past_7d'], df['receiver_fraud_rate_past_7d'] = calculate_target_encoding(df, 'receiver_account', '7d', alpha, global_rate)
    df['receiver_fraud_count_past_30d'], df['receiver_fraud_rate_past_30d'] = calculate_target_encoding(df, 'receiver_account', '30d', alpha, global_rate)
    
    # Device
    df['device_fraud_count_past_30d'], df['device_fraud_rate_past_30d'] = calculate_target_encoding(df, 'device_hash', '30d', alpha, global_rate)
    
    # IP
    df['ip_fraud_count_past_30d'], df['ip_fraud_rate_past_30d'] = calculate_target_encoding(df, 'ip_address', '30d', alpha, global_rate)
    
    # OOT Split
    train_inner = df.iloc[:3200000].copy()
    val = df.iloc[3200000:4000000].copy()
    
    # Hard Subset
    train_hard = train_inner[train_inner['tslt_is_missing'] == 0].copy()
    val_hard = val[val['tslt_is_missing'] == 0].copy()
    
    return train_inner, val, train_hard, val_hard

def evaluate_model(name, X_train, y_train, X_val, y_val, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    model = LGBMClassifier(scale_pos_weight=imbalance_ratio, n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    print(f"Training LightGBM on {name}...")
    pipeline.fit(X_train, y_train)
    
    probs_val = pipeline.predict_proba(X_val)[:, 1]
    probs_train = pipeline.predict_proba(X_train)[:, 1]
    
    pr_auc_val = average_precision_score(y_val, probs_val)
    pr_auc_train = average_precision_score(y_train, probs_train)
    roc_auc = roc_auc_score(y_val, probs_val)
    
    df_scores_val = pd.DataFrame({'score': probs_val, 'label': y_val}).sort_values('score', ascending=False)
    p01, _ = get_top_k(df_scores_val, y_val, 0.001)
    p1, r1 = get_top_k(df_scores_val, y_val, 0.01)
    p5, r5 = get_top_k(df_scores_val, y_val, 0.05)
    
    # Feature Importance Extraction
    classifier = pipeline.named_steps['classifier']
    preprocessor_pipe = pipeline.named_steps['preprocessor']
    ohe = preprocessor_pipe.named_transformers_['cat'].named_steps['onehot']
    ohe_cols = ohe.get_feature_names_out(cat_cols)
    all_cols = num_cols + list(ohe_cols)
    importances = classifier.feature_importances_
    df_imp = pd.DataFrame({'Feature': all_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    res = {
        'Dataset': name,
        'PR-AUC Val': pr_auc_val,
        'PR-AUC Train': pr_auc_train,
        'Gap': pr_auc_train - pr_auc_val,
        'ROC-AUC': roc_auc,
        'Prec@0.1%': p01,
        'Prec@1%': p1,
        'Prec@5%': p5,
        'Rec@1%': r1,
        'Rec@5%': r5
    }
    return res, df_imp

def main():
    write_md("# TARGET ENCODING EXPERIMENT REPORT (LAST CHANCE)\n", 'w')
    write_md("## A. Assumptions\n- Target history is calculated strictly on transactions **prior** to the current row.\n- Smoothed Rate Alpha = 10. Global Rate taken strictly from Train Inner (first 3.2M rows).\n- Production assumption: Fraud labels are known and available immediately for scoring (Zero Delay).\n")
    
    train_inner, val, train_hard, val_hard = prepare_data()
    
    base_num = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_negative']
    cat_cols = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    
    target_num = [
        'sender_fraud_count_past_7d', 'sender_fraud_rate_past_7d',
        'sender_fraud_count_past_30d', 'sender_fraud_rate_past_30d',
        'receiver_fraud_count_past_7d', 'receiver_fraud_rate_past_7d',
        'receiver_fraud_count_past_30d', 'receiver_fraud_rate_past_30d',
        'device_fraud_count_past_30d', 'device_fraud_rate_past_30d',
        'ip_fraud_count_past_30d', 'ip_fraud_rate_past_30d'
    ]
    
    results = []
    
    # 1. Full Validation
    full_num = base_num + ['tslt_is_missing'] + target_num
    res_full, imp_full = evaluate_model(
        "Full Validation", 
        train_inner[full_num + cat_cols], train_inner['is_fraud'].astype(int),
        val[full_num + cat_cols], val['is_fraud'].astype(int),
        full_num, cat_cols
    )
    results.append(res_full)
    
    # 2. Hard Subset
    hard_num = base_num + target_num
    res_hard, imp_hard = evaluate_model(
        "Hard Subset", 
        train_hard[hard_num + cat_cols], train_hard['is_fraud'].astype(int),
        val_hard[hard_num + cat_cols], val_hard['is_fraud'].astype(int),
        hard_num, cat_cols
    )
    results.append(res_hard)
    
    content = "## B. Evaluation Metrics\n"
    content += "| Dataset | Train PR-AUC | Val PR-AUC | Gap | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |\n"
    content += "|---|---|---|---|---|---|---|---|---|---|\n"
    
    for r in results:
        content += f"| {r['Dataset']} | {r['PR-AUC Train']:.5f} | {r['PR-AUC Val']:.5f} | {r['Gap']:.5f} | {r['ROC-AUC']:.5f} | {r['Prec@0.1%']*100:.2f}% | {r['Prec@1%']*100:.2f}% | {r['Prec@5%']*100:.2f}% | {r['Rec@1%']*100:.2f}% | {r['Rec@5%']*100:.2f}% |\n"
        
    write_md(content)
    
    content_imp = "## C. Top 15 Feature Importance (Full Validation Model)\n"
    content_imp += "| Rank | Feature | Importance |\n|---|---|---|\n"
    for i, r in imp_full.head(15).reset_index().iterrows():
        content_imp += f"| {i+1} | {r['Feature']} | {r['Importance']:.0f} |\n"
        
    write_md(content_imp)
    
    verdict = """
## D. Final Conclusions

1. **Target-history encoding có cải thiện không?**
   - Sự vươn lên của các biến `_fraud_count_` và `_fraud_rate_` trong Top Feature Importance sẽ trả lời cho câu hỏi này. Nếu PR-AUC Val tăng mạnh, nó khẳng định giả thuyết kẻ gian (hoặc IP/Device bị thỏa hiệp) thường xuyên tái phạm trong khung 7-30 ngày.
   
2. **Cải thiện có đủ lớn không?**
   - Đối chiếu PR-AUC Full Val với Baseline (0.04454). Nếu mức tăng chỉ xoay quanh 0.05-0.06, tín hiệu vẫn là Weak. Nếu đạt 0.10+, nó có Signal.
   
3. **Có overfit không?**
   - Quan sát cột **Gap (Train vs Val PR-AUC)**. Target-encoding cực kỳ nhạy cảm với việc bị học vẹt. Nếu Train PR-AUC > 0.5 nhưng Val PR-AUC lẹt đẹt ở 0.05, mô hình đã Overfit hoàn toàn và ghi nhớ mẫu một cách thảm họa.
   
4. **Feature này có hợp lệ trong production không?**
   - Dưới giả định "Không có độ trễ của nhãn lừa đảo" (Zero Delay), hệ thống này hợp lệ. Tuy nhiên, trong môi trường Ngân hàng thực tế, giao dịch lừa đảo thường mất 7-45 ngày mới bị phát hiện (thông qua Chargeback/Dispute). Việc dùng nhãn `past_7d` ở đây mang tính chất **Ảo tưởng Production** và sẽ sụp đổ khi triển khai thực tế.

5. **Có nên tiếp tục hay dừng?**
   - Nếu kết quả chỉ ra sự Overfit quá nặng hoặc PR-AUC không bứt phá, thì đã đến lúc đưa ra kết luận đóng băng: Tập dữ liệu này về bản chất thiếu tính năng phân tích (Predictive Features). Cần Data Generation mới hoặc từ chối thực hiện Modeling tiếp. KHÔNG được mở `test.csv`.
"""
    write_md(verdict)
    print("Target Encoding Experiment Completed!")

if __name__ == "__main__":
    main()
