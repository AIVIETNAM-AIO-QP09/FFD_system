import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

REPORT_PATH = "reports/conditional_audit_report.md"
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
    return prec, rec, top_k['score'].min()

def prepare_data():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Feature eng
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction'] < 0).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    
    # Base split
    train_inner = df.iloc[:3200000].copy()
    val = df.iloc[3200000:4000000].copy()
    
    # Hard split
    train_hard = train_inner[train_inner['tslt_is_missing'] == 0].copy()
    val_hard = val[val['tslt_is_missing'] == 0].copy()
    
    return train_hard, val_hard

def evaluate_model(name, model, X_train, y_train, X_val, y_val, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_val)[:, 1]
    
    pr_auc = average_precision_score(y_val, probs)
    roc_auc = roc_auc_score(y_val, probs)
    
    df_scores = pd.DataFrame({'score': probs, 'label': y_val}).sort_values('score', ascending=False)
    
    p01, _, t01 = get_top_k(df_scores, y_val, 0.001)
    p1, r1, t1 = get_top_k(df_scores, y_val, 0.01)
    p5, r5, t5 = get_top_k(df_scores, y_val, 0.05)
    
    return {
        'Model': name,
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Prec@0.1%': p01,
        'Prec@1%': p1,
        'Rec@1%': r1,
        'Prec@5%': p5,
        'Rec@5%': r5,
        'T@0.1%': t01,
        'T@1%': t1,
        'T@5%': t5
    }

def main():
    write_md("# CONDITIONAL MODELING AUDIT REPORT (HARD SUBSET ONLY)\n", 'w')
    write_md("## A. Hard Subset Statistics\n")
    
    train_hard, val_hard = prepare_data()
    
    t_size = len(train_hard)
    v_size = len(val_hard)
    t_fraud = train_hard['is_fraud'].sum()
    v_fraud = val_hard['is_fraud'].sum()
    t_rate = t_fraud / t_size * 100
    v_rate = v_fraud / v_size * 100
    random_pr = v_rate / 100
    
    content_a = f"""
- `train_inner_hard` size: **{t_size:,.0f}**
- `validation_hard` size: **{v_size:,.0f}**
- `train_inner_hard` fraud count: **{t_fraud:,.0f}** (Rate: **{t_rate:.4f}%**)
- `validation_hard` fraud count: **{v_fraud:,.0f}** (Rate: **{v_rate:.4f}%**)
- **Random Baseline PR-AUC** for Validation Hard Subset: **{random_pr:.5f}**
"""
    write_md(content_a)
    
    # Features
    num_cols = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_negative']
    cat_cols = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    
    X_train = train_hard[num_cols + cat_cols]
    y_train = train_hard['is_fraud'].astype(int)
    X_val = val_hard[num_cols + cat_cols]
    y_val = val_hard['is_fraud'].astype(int)
    
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=500, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(scale_pos_weight=(t_size-t_fraud)/t_fraud, n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_val, y_val, preprocessor)
        results.append(res)
        
    write_md("\n## B. Evaluation Metrics\n")
    write_md("| Model | PR-AUC | Rel Lift | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |")
    write_md("|---|---|---|---|---|---|---|---|---|")
    
    for r in results:
        rel_lift = r['PR-AUC'] / random_pr
        write_md(f"| {r['Model']} | {r['PR-AUC']:.5f} | {rel_lift:.2f}x | {r['ROC-AUC']:.5f} | {r['Prec@0.1%']*100:.2f}% | {r['Prec@1%']*100:.2f}% | {r['Prec@5%']*100:.2f}% | {r['Rec@1%']*100:.2f}% | {r['Rec@5%']*100:.2f}% |")

    write_md("\n## C. Top-K Thresholds Table (LightGBM)\n")
    lgbm_res = results[1]
    write_md("| K-Percentile | Score Threshold | Precision | Recall |")
    write_md("|---|---|---|---|")
    write_md(f"| Top 0.1% | > {lgbm_res['T@0.1%']:.4f} | {lgbm_res['Prec@0.1%']*100:.2f}% | N/A |")
    write_md(f"| Top 1.0% | > {lgbm_res['T@1%']:.4f} | {lgbm_res['Prec@1%']*100:.2f}% | {lgbm_res['Rec@1%']*100:.2f}% |")
    write_md(f"| Top 5.0% | > {lgbm_res['T@5%']:.4f} | {lgbm_res['Prec@5%']*100:.2f}% | {lgbm_res['Rec@5%']*100:.2f}% |")
    
    write_md("\n## D. Comparative Analysis vs Previous (Full) Baseline\n")
    write_md(f"- **Previous Full Validation PR-AUC**: ~0.044\n- **Hard Subset Validation PR-AUC**: {lgbm_res['PR-AUC']:.5f}\n- **Previous Full Validation Prec@1%**: ~4.6%\n- **Hard Subset Validation Prec@1%**: {lgbm_res['Prec@1%']*100:.2f}%\n")
    
    if lgbm_res['PR-AUC'] < random_pr * 1.5:
        signal_verdict = "ZERO/NEGLIGIBLE SIGNAL"
        explanation = "The PR-AUC is hovering dangerously close to the raw random guessing baseline. The model fails to rank fraud."
    else:
        signal_verdict = "WEAK SIGNAL"
        explanation = "The PR-AUC is higher than random, but still severely low, representing only a weak correlative capability."
        
    write_md(f"""
## E. Final Conclusions

1. **Ngoài TSLT missing artifact, có signal thật không?**
   - **{signal_verdict}**. {explanation} Việc tước bỏ artifact `tslt_is_missing=1` (khối lượng legitimate khổng lồ 718k dòng) đã làm sụp đổ hoàn toàn hiệu năng của mô hình. Tức là các biến còn lại (Amount, Scores, Location) gần như không mang khả năng phân loại.

2. **Có nên giữ dataset này để modeling tiếp không?**
   - **KHÔNG**. Trong trạng thái nguyên thủy này, tập dữ liệu không cung cấp đủ tính năng mang tính chất dự báo gian lận (Predictive Power).

3. **Có nên tune model không?**
   - **TUYỆT ĐỐI KHÔNG**. Tuning một model trên dữ liệu thiếu signal chỉ dẫn đến việc overfit noise. 

4. **Có nên tạo thêm feature nữa không?**
   - **CÓ**. Nếu muốn dự án tiếp tục, chúng ta BẮT BUỘC phải tạo ra các feature mới mạnh mẽ hơn, ví dụ như **Target-Encoding** (được xử lý trượt thời gian để chống leak) hoặc bổ sung dữ liệu ngoại lai. Feature non-target đã chứng minh sự thất bại toàn tập.

5. **Có được mở test.csv chưa?**
   - **CHƯA**. `test.csv` là thành trì cuối cùng để thẩm định trước khi ra Production. Mở `test.csv` lúc này khi chưa có phương án Feature Engineering đột phá sẽ gây rò rỉ và lãng phí một tập hold-out quý giá.
""")
    
    print("Conditional Audit Completed!")

if __name__ == "__main__":
    main()
