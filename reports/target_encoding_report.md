# TARGET ENCODING EXPERIMENT REPORT (LAST CHANCE)

## A. Assumptions
- Target history is calculated strictly on transactions **prior** to the current row.
- Smoothed Rate Alpha = 10. Global Rate taken strictly from Train Inner (first 3.2M rows).
- Production assumption: Fraud labels are known and available immediately for scoring (Zero Delay).

## B. Evaluation Metrics
| Dataset | Train PR-AUC | Val PR-AUC | Gap | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|---|---|---|
| Full Validation | 0.05880 | 0.04398 | 0.01481 | 0.59303 | 5.38% | 4.17% | 4.34% | 1.16% | 6.01% |
| Hard Subset | 0.06048 | 0.04414 | 0.01634 | 0.50055 | 4.73% | 4.71% | 4.36% | 1.07% | 4.95% |

## C. Top 15 Feature Importance (Full Validation Model)
| Rank | Feature | Importance |
|---|---|---|
| 1 | spending_deviation_score | 421 |
| 2 | tslt_abs | 394 |
| 3 | amount | 329 |
| 4 | geo_anomaly_score | 298 |
| 5 | hour | 217 |
| 6 | velocity_score | 196 |
| 7 | month | 116 |
| 8 | day_of_week | 110 |
| 9 | sender_fraud_rate_past_30d | 81 |
| 10 | tslt_is_missing | 76 |
| 11 | receiver_fraud_rate_past_30d | 60 |
| 12 | payment_channel_UPI | 33 |
| 13 | payment_channel_ACH | 32 |
| 14 | payment_channel_wire_transfer | 29 |
| 15 | device_fraud_rate_past_30d | 28 |


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

