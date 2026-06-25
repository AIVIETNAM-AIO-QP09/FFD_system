# CONDITIONAL MODELING AUDIT REPORT (HARD SUBSET ONLY)

## A. Hard Subset Statistics


- `train_inner_hard` size: **2,625,638**
- `validation_hard` size: **656,284**
- `train_inner_hard` fraud count: **114,643** (Rate: **4.3663%**)
- `validation_hard` fraud count: **28,884** (Rate: **4.4011%**)
- **Random Baseline PR-AUC** for Validation Hard Subset: **0.04401**


## B. Evaluation Metrics

| Model | PR-AUC | Rel Lift | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.04416 | 1.00x | 0.50050 | 4.42% | 4.36% | 4.31% | 0.99% | 4.89% |
| LightGBM | 0.04407 | 1.00x | 0.50052 | 4.27% | 4.57% | 4.34% | 1.04% | 4.93% |

## C. Top-K Thresholds Table (LightGBM)

| K-Percentile | Score Threshold | Precision | Recall |
|---|---|---|---|
| Top 0.1% | > 0.5561 | 4.27% | N/A |
| Top 1.0% | > 0.5346 | 4.57% | 1.04% |
| Top 5.0% | > 0.5268 | 4.34% | 4.93% |

## D. Comparative Analysis vs Previous (Full) Baseline

- **Previous Full Validation PR-AUC**: ~0.044
- **Hard Subset Validation PR-AUC**: 0.04407
- **Previous Full Validation Prec@1%**: ~4.6%
- **Hard Subset Validation Prec@1%**: 4.57%


## E. Final Conclusions

1. **Ngoài TSLT missing artifact, có signal thật không?**
   - **ZERO/NEGLIGIBLE SIGNAL**. The PR-AUC is hovering dangerously close to the raw random guessing baseline. The model fails to rank fraud. Việc tước bỏ artifact `tslt_is_missing=1` (khối lượng legitimate khổng lồ 718k dòng) đã làm sụp đổ hoàn toàn hiệu năng của mô hình. Tức là các biến còn lại (Amount, Scores, Location) gần như không mang khả năng phân loại.

2. **Có nên giữ dataset này để modeling tiếp không?**
   - **KHÔNG**. Trong trạng thái nguyên thủy này, tập dữ liệu không cung cấp đủ tính năng mang tính chất dự báo gian lận (Predictive Power).

3. **Có nên tune model không?**
   - **TUYỆT ĐỐI KHÔNG**. Tuning một model trên dữ liệu thiếu signal chỉ dẫn đến việc overfit noise. 

4. **Có nên tạo thêm feature nữa không?**
   - **CÓ**. Nếu muốn dự án tiếp tục, chúng ta BẮT BUỘC phải tạo ra các feature mới mạnh mẽ hơn, ví dụ như **Target-Encoding** (được xử lý trượt thời gian để chống leak) hoặc bổ sung dữ liệu ngoại lai. Feature non-target đã chứng minh sự thất bại toàn tập.

5. **Có được mở test.csv chưa?**
   - **CHƯA**. `test.csv` là thành trì cuối cùng để thẩm định trước khi ra Production. Mở `test.csv` lúc này khi chưa có phương án Feature Engineering đột phá sẽ gây rò rỉ và lãng phí một tập hold-out quý giá.

