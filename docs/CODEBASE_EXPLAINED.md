# Codebase Explained — Financial Fraud Detection & Fraud Review Co-pilot

> For a visual teammate onboarding guide, see `docs/TEAM_ONBOARDING_VISUAL_GUIDE.md`.

## 1. Project Overview
* Đây là project fraud detection trên financial transaction dataset.
* Mục tiêu chính là xây dựng weak but validated fraud-ranking baseline.
* Mục tiêu mở rộng là tạo Fraud Review Co-pilot hỗ trợ analyst bằng LLM/RAG.
* Final ML model không phải production-ready autonomous fraud detector.
* LLM không đưa ra quyết định fraud, chỉ hỗ trợ tạo analyst-facing brief.

## 2. High-Level Architecture
Pipeline tổng thể của dự án diễn ra theo trình tự sau:

```text
Raw transaction data
→ Data split
→ EDA
→ Leakage audit
→ Feature engineering
→ Model training
→ Validation
→ Final locked holdout test
→ Error/limitation analysis
→ Graph feasibility audit
→ Cascade V2 research
→ Evidence builder
→ Similar-case retriever
→ Local Qwen3-8B candidate wording
→ Validator-first deterministic guardrail
→ Analyst-facing fraud investigation brief
```

## 3. Repository Structure

```text
project-root/
├── configs/            # Chứa các file cấu hình môi trường hoặc pipeline
├── data/               # Thư mục lưu dữ liệu raw và dữ liệu đã xử lý
│   └── split/          # Dữ liệu được chia thành train/test
├── docs/               # Thư mục chứa tài liệu kỹ thuật
├── models/             # Chứa các model ML, retriever và preprocessor đã được serialize (.pkl)
├── notebooks/          # Chứa các Jupyter Notebooks dùng để research và thử nghiệm (Baseline.ipynb, chain-of-thought-v1-04.ipynb)
├── reports/            # Chứa toàn bộ các báo cáo kết quả từ quá trình EDA, feature engineering, model training, evaluation và LLM sample briefs
├── src/                # Mã nguồn chính chứa các module preprocessing, training, evaluation, và các run script (ví dụ: run_feature_engineering.py, run_baseline_modeling.py)
├── tests/              # Thư mục chứa test scripts (unit tests)
├── README.md           # Giới thiệu cơ bản về project
├── requirements.txt    # Danh sách các thư viện cần cài đặt
├── run_lmstudio_copilot_demo.py # Script chính để chạy LLM Co-pilot
├── run_pipeline.py     # Script chạy pipeline tổng thể
└── split_data.py       # Script thực hiện chia dữ liệu ban đầu
```

## 4. Data Files and Split Protocol
* Dữ liệu train và test được lưu tại `data/split/train.csv` và `data/split/test.csv`.
* Train có 4,000,000 transactions.
* Test có 1,000,000 transactions.
* Train/test split được thực hiện dựa trên timestamp (sử dụng file `split_data.py`).
* Train inner/validation split:
  * train_inner: first 3.2M rows
  * validation: last 800K rows
* Test là final holdout và chỉ được mở sau khi lock pipeline hoàn toàn.
* Không dùng random split vì đây là bài toán time-series-like liên quan đến fraud.

## 5. Target and Leakage Control
Để đảm bảo tính trung thực cho model evaluation, các nguyên tắc sau được kiểm soát nghiêm ngặt:
* Cột `fraud_type` là direct target leakage và không được dùng làm feature. Việc loại bỏ cột này được thực hiện trong quá trình tiền xử lý ở `src/preprocessing.py`.
* Cột `transaction_id` không dùng làm feature.
* Raw high-cardinality IDs như `sender_account`, `receiver_account`, `ip_address`, `device_hash` không được dùng trực tiếp làm feature.
* Không sử dụng target encoding trong final model.
* Không dùng label-history features trong final model.
* Không dùng SMOTE trong main pipeline.
* Các historical/novelty features phải là strictly-past (chỉ nhìn về quá khứ, không nhìn dữ liệu tương lai).

## 6. EDA Summary
Các phát hiện EDA quan trọng (báo cáo chi tiết có tại `reports/eda_train_report.md` và `reports/eda_train_deep_dive_report.md`):
* Fraud rate khoảng 3.59% trong tập train.
* Amount distribution giữa fraud và legitimate gần như giống nhau.
* Low-cardinality categorical features có lift yếu.
* Behavioral scores có signal yếu.
* Raw IDs là high-cardinality và không an toàn nếu dùng trực tiếp.
* `time_since_last_transaction` có artifact lớn.
* Các giao dịch có `tslt_is_missing = 1` có near-zero/zero fraud trong tập train/validation, dẫn đến việc model phụ thuộc mạnh vào artifact này.

## 7. Feature Engineering
Các logic feature engineering được quản lý chủ yếu trong `src/run_feature_engineering.py` và module preprocessor.

### 7.1 Baseline Features
* `amount`
* `log_amount`
* `spending_deviation_score`
* `velocity_score`
* `geo_anomaly_score`
* `hour`
* `day_of_week`
* `month`
* `is_weekend`
* `is_night`
* TSLT variants:
  * `tslt_abs`
  * `tslt_is_missing`
  * `tslt_is_negative`

### 7.2 Categorical Features
* `transaction_type`
* `merchant_category`
* `payment_channel`
* `device_used`
* `location`

### 7.3 Novelty Features
Được tính toán chặt chẽ từ lịch sử quá khứ, hoàn toàn không dính target leakage:
* `is_new_location_for_sender`
* `is_new_payment_channel_for_sender`
* `is_new_transaction_type_for_sender`
* `is_new_device_used_for_sender`

### 7.4 Rejected / Non-final Features
* SMOTE: gây nhiễu phân phối thật.
* Target-history features và Target encoding: nguy cơ data leakage trong môi trường production cao.
* Raw IDs: quá nhiều giá trị (high-cardinality), dễ dẫn đến overfitting.
* Graph deep features & Node2Vec/GNN: tín hiệu yếu, không mang lại cải thiện đủ lớn để bù đắp chi phí triển khai.
* Các feature historical rolling (24h/7d): dữ liệu quá sparse.

## 8. Final Model Pipeline
Final feature set: **FS1 — Baseline + Novelty Binary**.

**Numeric features:**
```text
amount
log_amount
spending_deviation_score
velocity_score
geo_anomaly_score
hour
day_of_week
month
is_weekend
is_night
tslt_abs
tslt_is_missing
tslt_is_negative
is_new_location_for_sender
is_new_payment_channel_for_sender
is_new_transaction_type_for_sender
is_new_device_used_for_sender
```

**Categorical features:**
```text
transaction_type
merchant_category
payment_channel
device_used
location
```

**Preprocessing:** (Định nghĩa trong `src/preprocessing.py`)
* Numeric: `SimpleImputer(median)`
* Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`

**Model:** (Định nghĩa và train trong `src/run_baseline_modeling.py` / `src/train_and_export.py`)
```python
LightGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=300,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    random_state=42
)
```

## 9. Model Evaluation
Đánh giá final model trên tập holdout test (báo cáo tại `reports/final_test_evaluation_report.md`):

```text
PR-AUC: 0.04819
ROC-AUC: 0.62329
Precision@0.1%: 4.60%
Precision@0.5%: 5.28%
Precision@1%: 4.78%
Precision@2%: 4.95%
Precision@5%: 4.94%
Precision@10%: 4.88%
Recall@1%: 1.33%
Recall@5%: 6.86%
Recall@10%: 13.56%
Lift@1%: 1.33x
Lift@5%: 1.37x
Lift@10%: 1.36x
```

**Diễn giải:**
* Random baseline PR-AUC xấp xỉ fraud rate test: 3.6026%.
* Model tốt hơn random.
* Signal có thật nhưng yếu.
* Không đủ cho automatic fraud blocking.
* Phù hợp hơn với review prioritization (sắp xếp thứ tự ưu tiên kiểm tra thủ công).

## 10. Business Interpretation
* Nếu bộ phận vận hành tiến hành review top 1% test = 10,000 transactions.
* Nếu review random sẽ bắt được khoảng 360 case fraud.
* Nếu review dựa trên ranking của model sẽ bắt được 478 case fraud.
* Tăng thêm khoảng 118 fraud cases so với random.
* Đây là gain có ích cho manual review queue, tuy nhiên precision vẫn còn ở mức thấp.

## 11. Graph Analysis / Graph Feasibility Audit
Báo cáo kiểm tra tính khả thi của graph tại `reports/graph_feasibility_audit_report.md` (chạy bởi `src/run_graph_audit.py`):
* Dữ liệu có nhiều node/edge nhưng repeated fraud-relevant connectivity khá yếu.
* Việc áp dụng sender/receiver/device graph features chỉ cải thiện rất ít hoặc cho kết quả gần giống random.
* Hướng Node2Vec/GNN không đáng để triển khai sâu trên dataset hiện tại.
* Kết luận: Graph feature không phải là hướng đi chính thức cho final model.

## 12. Cascade V2 Research
Nghiên cứu về mô hình Cascade B (báo cáo tại `reports/segmented_cascade_experiment_report.md`, chạy bởi `src/run_cascade_experiment.py`):
* Cascade B là ứng viên (V2 candidate) cho tương lai.
* Validation PR-AUC tăng nhẹ từ khoảng 0.04882 lên 0.04934.
* Improvement này quá nhỏ, chưa đủ lớn để thay thế final tested model.
* Quyết định: Giữ lại làm research/productization candidate, nhưng không thay đổi final holdout model.

## 13. LLM/RAG Fraud Review Co-pilot
Productization extension hỗ trợ việc giải thích nguyên nhân rủi ro cho analyst.

**Architecture:**
```text
Scored transaction
→ Evidence builder
→ Similar-case retriever
→ Qwen3-8B candidate wording via LM Studio
→ Validator-first deterministic guardrail layer
→ Fraud Investigation Brief
```

### 13.1 Local LLM Setup
Tích hợp model thông qua class LLM Client tại `src/llm_client.py`.
* Runtime: LM Studio local server
* Model: Qwen3-8B GGUF Q4_K_M
* Endpoint: `http://127.0.0.1:1234/v1`
* Model identifier: `qwen/qwen3-8b:2`
* Temperature: 0.1
* Purpose: Generate candidate analyst wording

### 13.2 Similar-case Retriever
* Retriever dùng structured nearest neighbors dựa trên không gian feature của giao dịch.
* Similar cases chỉ đóng vai trò là nearest-neighbor examples.
* Không phải semantic case-note RAG thật.
* Không mang tính chất chứng minh (causal proof).

### 13.3 Guardrail Design
* Output của Qwen3 chỉ được coi là candidate wording ban đầu.
* Tất cả unsupported risk language đều bị chặn.
* Báo cáo cuối cùng (final analyst-facing report) sử dụng deterministic guarded text (các template đã được chốt từ trước để đảm bảo an toàn).
* LLM không đưa ra quyết định giao dịch là fraud hay legit.
* LLM không đưa ra đề xuất auto-block, freeze funds, hay suspend account.

### 13.4 Final Brief Format
Báo cáo đầu ra cho analyst sẽ bao gồm các phần cố định:
* Transaction ID
* Model Risk Score
* Risk Band
* Recommended Action
* Key Risk Factors
* Similar Historical Cases
* Why This Was Flagged / Why This Was Not Escalated
* Suggested Analyst Next Step
* Caveats

### 13.5 Final Sample Report
File chứa các sample brief đã tạo thành công được lưu tại:
`reports/lmstudio_qwen3_copilot_samples_final_deterministic.md`
Sample gồm:
* 2 High cases
* 2 Review cases
* 2 Medium cases
* 2 Low cases

## 14. Important Source Files

| Path | Purpose | Notes |
| ---- | ------- | ----- |
| `split_data.py` | Chia raw data thành train.csv và test.csv dựa theo timestamp | Tại root directory |
| `src/run_eda_deep_dive.py` | Script phân tích EDA chuyên sâu | Lưu báo cáo đầu ra vào thư mục `reports/` |
| `src/preprocessing.py` | Logic xử lý dữ liệu và định nghĩa Preprocessor | Tiền xử lý null, one-hot encoding |
| `src/run_feature_engineering.py` | Tạo baseline và novelty features | |
| `src/run_baseline_modeling.py` | Huấn luyện các baseline models bằng FS1 | Thử nghiệm LightGBM, Random Forest |
| `src/run_final_evaluation.py` | Đánh giá final model trên tập holdout test | Mở test set 1 lần duy nhất |
| `src/run_graph_audit.py` | Đánh giá tính hiệu quả của Graph features (Node2Vec) | Cho thấy tín hiệu không đủ mạnh |
| `src/run_cascade_experiment.py` | Thử nghiệm mô hình Cascade B (V2 candidate) | Phân mảnh data để mô hình hoá |
| `src/train_and_export.py` | Huấn luyện final model và xuất files `.pkl` | Lưu vào thư mục `models/` |
| `src/llm_client.py` | Giao tiếp API nội bộ với LM Studio và áp dụng Guardrails | Chặn các cụm từ risk bị cấm |
| `run_lmstudio_copilot_demo.py` | Script tạo Fraud Brief Report sử dụng mô hình LLM + RAG | Khởi tạo Validator-first framework |
| `reports/final_project_report.md`| Báo cáo tổng hợp cuối cùng toàn bộ kết quả dự án | Report tổng |
| `README.md` | Tài liệu giới thiệu tổng quan hệ thống repo | File hướng dẫn ở mức gốc |

## 15. How to Run

### 15.1 Setup environment
* Cài đặt môi trường Python.
* Cài đặt các thư viện phụ thuộc bằng pip:
```bash
pip install -r requirements.txt
```

### 15.2 Run ML pipeline
Chạy các bước chuẩn bị dữ liệu và huấn luyện theo trình tự chuẩn (hoặc dùng `python run_pipeline.py`):
```bash
python split_data.py
python src/run_eda_deep_dive.py
python src/run_feature_engineering.py
python src/run_baseline_modeling.py
python src/train_and_export.py
python src/run_final_evaluation.py
```

### 15.3 Run LM Studio Co-pilot Demo
1. Mở phần mềm LM Studio trên máy tính.
2. Load model **Qwen3-8B GGUF Q4_K_M** (định danh: `qwen/qwen3-8b:2`).
3. Khởi động **Local Server** trên LM Studio.
4. Đảm bảo server endpoint đang hoạt động tại:
```text
http://127.0.0.1:1234/v1
```
5. Chạy file script để lấy output demo:
```bash
python run_lmstudio_copilot_demo.py
```

## 16. Generated Artifacts
Các kết quả và object đã sinh ra sau khi chạy quá trình trên:
* **reports/**: `final_project_report.md`, `lmstudio_qwen3_copilot_samples_final_deterministic.md`, `eda_train_deep_dive_report.md`, `feature_engineering_report.md`, `graph_feasibility_audit_report.md`, v.v.
* **models/**: `calibrated_lgbm.pkl`, `preprocessor.pkl`, `retriever.pkl`, `history_store.pkl`, `rag_corpus.pkl`.
* **notebooks/**: Các Jupyter Notebook `Baseline.ipynb`, `chain-of-thought-v1-04.ipynb` chứa toàn bộ log exploration thô.

## 17. Known Limitations
* Tín hiệu phân loại (signal) khá yếu; các chỉ số đo lường thực tế PR-AUC và Precision@K thấp.
* Artifact missing của biến TSLT gây ảnh hưởng quá lớn (hiện tượng thiếu `time_since_last_transaction` đi kèm fraud zero).
* Dữ liệu cục bộ hoàn toàn bị thiếu external enrichment (VD: IP blacklist, third-party scoring).
* Không có thông tin real device footprint hoặc IP reputation.
* Trải nghiệm truy vấn không phải là real semantic RAG do thiếu analyst note ghi chú thủ công từ con người.
* Không có production feedback loop thu thập dữ liệu nhãn trả về tự động sau khi chuyên viên đánh giá.
* Hệ thống này KHÔNG ĐƯỢC dùng để đưa ra quyết định đóng băng tài khoản hay block giao dịch tự động.
* LLM hoàn toàn không tham gia nâng cao độ chính xác của phân loại fraud, mục đích chính chỉ là diễn giải kết quả.
* Mọi cố gắng improvement thêm yêu cầu cung cấp data tốt hơn hoặc phải holdout một bộ test set mới do test set hiện tại đã được đánh giá một lần (pipeline locked).

## 18. Monitoring Recommendations
Dưới đây là một số khuyến nghị khi triển khai đo đạc thực tế:
* Theo dõi Fraud rate drift.
* Theo dõi Score distribution drift để giám sát điểm rank của model.
* Đánh giá Precision@K và Recall@K theo mốc thời gian thực tế.
* Theo dõi sự thay đổi TSLT missing rate drift.
* Theo dõi Review queue volume – tức lượng giao dịch được cảnh báo và xếp hàng chờ chuyên viên phê duyệt.
* Đo lường Analyst override rate – tần suất chuyên viên thay đổi/phủ nhận phán đoán của model.
* Đếm tần suất các trường hợp hệ thống LLM guardrail failure (LLM trả về nội dung vi phạm policy và bị hệ thống phát hiện chặn).
* Đếm tần suất fallback usage (sử dụng format mẫu deterministic thay vì text sinh ra từ LLM).

## 19. Final Project Positioning
* Mô hình cung cấp một **weak but validated fraud-ranking baseline**.
* Model được thiết kế để hỗ trợ **manual review prioritization** (ưu tiên duyệt giao dịch cho nhân sự con người).
* Fraud Review Co-pilot (LLM/RAG) đóng vai trò giúp chuẩn hóa và diễn giải bản báo cáo investigation brief, tối ưu quy trình phân tích của analyst.
* Toàn bộ hệ thống không phải một autonomous fraud detector.
* Giá trị cốt lõi đúc kết từ project là việc thiết lập được một pipeline end-to-end đúng đắn và chặt chẽ: leakage-aware modeling + honest validation methodology + productization-safe LLM assistant layer.

## 20. Appendix: Glossary
* **PR-AUC**: Precision-Recall Area Under Curve. Thước đo hiệu suất quan trọng cho các bài toán mất cân bằng class dữ liệu trầm trọng như Fraud Detection.
* **ROC-AUC**: Receiver Operating Characteristic Area Under Curve.
* **Precision@K**: Tỷ lệ giao dịch thật sự là fraud trong số top K% giao dịch bị model chấm điểm rủi ro cao nhất.
* **Recall@K**: Tỷ lệ số fraud model bắt được nằm trong top K% so với tổng số vụ fraud tồn tại thực tế.
* **Lift@K**: Độ hiệu quả của model ở top K% so với việc chọn review ngẫu nhiên một tỷ lệ giao dịch tương tự.
* **TSLT**: Time Since Last Transaction.
* **Leakage**: Sự rò rỉ dữ liệu hoặc nhãn từ tương lai vào quá trình huấn luyện model khiến kết quả over-optimistic.
* **OOT validation**: Out-Of-Time validation. Dùng dữ liệu quá khứ train mô hình để dự đoán khoảng thời gian sau đó nhằm kiểm chứng model.
* **Holdout test**: Tập dữ liệu riêng biệt không bao giờ sử dụng trong quá trình huấn luyện, chỉ được dùng đánh giá lần cuối.
* **RAG**: Retrieval-Augmented Generation, việc đính kèm dữ liệu liên quan vào context nhằm định hướng nội dung của LLM.
* **Nearest-neighbor retrieval**: Tìm kiếm và lấy ra các giao dịch lân cận gần nhất trong không gian feature vector.
* **Guardrail**: Cơ chế phòng vệ bằng rules cứng để kiểm duyệt output sinh ra của LLM trước khi hiển thị cho người dùng.
* **Deterministic fallback**: Đoạn code / string template tĩnh thay thế nội dung của LLM khi xảy ra vi phạm Guardrail.
* **Weak ranking signal**: Khả năng phân loại (xếp hạng rủi ro) chỉ nhỉnh hơn mức ngẫu nhiên nhưng chưa đủ mạnh để tự động hóa toàn bộ.
