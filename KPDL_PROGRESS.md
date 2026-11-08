# Kết quả thực nghiệm và trạng thái triển khai

## Pipeline đã triển khai

1. Làm sạch giao dịch, lưu báo cáo số dòng bị loại và dữ liệu thiếu.
2. Tổng hợp nhu cầu theo sản phẩm/ngày, điền ngày không bán bằng 0.
3. Phân tích EDA, tạo bảng và biểu đồ theo thời gian/sản phẩm.
4. K-Means với bốn đặc trưng sản phẩm; chuẩn hóa và đánh giá Elbow/Silhouette.
5. Apriori trên từng hóa đơn; báo cáo luật theo support, confidence, lift.
6. Phát hiện bất thường tổng số lượng và doanh thu theo ngày bằng ngưỡng rolling median và Isolation Forest.
7. Feature engineering có lag/rolling dịch ít nhất một ngày; chia theo thời gian; cụm sản phẩm chỉ fit từ train.
8. So sánh Linear Regression, Random Forest, HistGradientBoosting; chọn trên validation và đánh giá test riêng.

## Xử lý các tồn đọng trong checklist

- EDA bổ sung thống kê doanh thu, số hóa đơn và lượng mua theo khách hàng/quốc gia; xuất `customer_summary.csv`, `country_summary.csv` và hai biểu đồ phân bố/top quốc gia.
- Apriori xuất toàn bộ 5.187 luật vào JSON/CSV (không chỉ top 100), kèm diễn giải top luật theo lift, confidence và support trong `association_interpretation.md`.
- Anomaly product/day nay chỉ xét ngày có bán, dựa vào các ngày bán trước đó trong cửa sổ 90 ngày và hàng rào Q3 + 1,5×IQR; không đánh dấu ngày không bán. Lần chạy hiện tại đánh dấu 41.168/530.098 (7,766%) ngày bán để rà soát, không tự xóa những dòng này. Ngưỡng này vẫn là heuristic cần chuyên gia nghiệp vụ xác nhận.
- Dự báo bổ sung mô hình hai giai đoạn (xác suất có bán × lượng bán khi có bán), chọn cutoff bằng validation. Cutoff được chọn là 0,55. Kết quả test: MAE 3,881; RMSE 24,793; WAPE 1,078; R² 0,134. Tốt hơn baseline Lag-1 (WAPE 1,454; R² -0,685), nhưng WAPE vẫn lớn hơn 1 nên chưa đủ cơ sở khuyến nghị vận hành.

## Thảo luận và hạn chế của mô hình

DailyDemand có tính gián đoạn cao: khoảng 78–80% quan sát sản phẩm-ngày bằng 0. Vì vậy mô hình được tách thành hai phần: ước lượng xác suất phát sinh bán hàng và ước lượng số lượng có điều kiện ngày có bán; dự báo cuối dùng cutoff 0,55 chọn trên validation. Trên test, TwoStageDemand đạt MAE 3,88, RMSE 24,79 và R² 0,134, tốt hơn baseline Lag-1 (MAE 5,23, RMSE 34,59, R² -0,685). Kết quả cho thấy mô hình khai thác được tín hiệu tốt hơn baseline đơn giản.

Tuy nhiên, WAPE test là 1,078 (>1), tức tổng sai số tuyệt đối vẫn lớn hơn tổng nhu cầu thực. R² 0,134 cũng cho thấy phần biến thiên được giải thích còn hạn chế. Do đó không kết luận mô hình đã hoàn hảo hoặc sẵn sàng tự động quyết định tồn kho. Kết quả hiện phù hợp để minh họa quy trình và hỗ trợ phân tích; cần đánh giá thêm theo nhóm sản phẩm, thời gian dự báo, chi phí thiếu hàng/tồn kho và rolling-origin validation trước khi triển khai.

## Kết quả lần chạy hiện tại

- Dữ liệu sạch: 1.007.913 dòng; 4.745 mã hàng; 5.878 khách hàng; 43 quốc gia.
- 10% sản phẩm đứng đầu chiếm khoảng 63,17% doanh thu.
- K-Means: k=3, Silhouette=0,5464 trên 3.271 sản phẩm đủ lịch sử train.
- Apriori: support tối thiểu 0,5%, confidence tối thiểu 15%, sinh 5.187 luật; toàn bộ luật được lưu trong JSON/CSV, cùng phần diễn giải luật tiêu biểu.
- Bất thường: 30 ngày theo Quantity và 29 ngày theo Revenue theo quy tắc rolling; 13 ngày theo Isolation Forest.
- Mô hình được chọn trên validation: TwoStageDemand, cutoff 0,55 (MAE 2,9081; RMSE 26,0989; WAPE 0,9491; R² 0,0555).
- Test: MAE 3,8807; RMSE 24,7927; WAPE 1,0782; R² 0,1343. Baseline Lag-1: MAE 5,2342; RMSE 34,5910; WAPE 1,4543; R² -0,6852.

## Diễn giải và giới hạn

Pipeline đã chạy đủ các nhánh kỹ thuật chính nêu trong đề cương. TwoStageDemand cải thiện so với baseline Lag-1, nhưng WAPE test vẫn lớn hơn 1; do đó chưa nên kết luận mô hình có thể dùng trực tiếp để lập tồn kho. Các hướng tiếp theo phù hợp là dự báo riêng theo nhóm sản phẩm, đánh giá theo chân trời dự báo khác nhau, thêm chỉ số MASE/SMAPE và hàm chi phí tồn kho, rồi thực hiện rolling-origin validation.

XGBoost không được cài trong môi trường, nên pipeline dùng HistGradientBoosting làm mô hình boosting thay thế. Nếu yêu cầu bắt buộc đúng XGBoost theo sơ đồ, cần bổ sung dependency và chạy lại so sánh.

## Chạy toàn bộ

```bash
uv sync
uv run python preprocess.py
uv run python eda.py
uv run python feature_engineering.py
uv run python clustering.py
uv run python mining.py
uv run python modeling.py

# 3 phân hệ nâng cấp mới:
uv run python intermittent_models.py
uv run python weekly_forecasting.py
uv run python inventory_cost.py
```

## Kết quả thực nghiệm 3 phân hệ nâng cấp mới (sau kiểm toán QA & hiệu chỉnh)

### 1. Mô hình Croston, SBA & TSB (Teunter-Syntetos-Babai)
- Tối ưu tham số trên validation (trên đúng 3.271 sản phẩm chuẩn): Croston $\alpha=0.05$; SBA $\alpha=0.20$ (hệ số hiệu chỉnh $(1-\alpha/2)=0.90$); TSB $\alpha=0.20, \beta=0.40$.
- **Kết quả tập Test (928.964 quan sát):**
  - **TwoStageDemand (Hurdle ML đa biến):** MAE 3.8807, RMSE 24.7927, **WAPE 1.0782**, $R^2$ **0.1343**.
  - **TSB (Đơn biến):** MAE 4.5276, RMSE 25.7234, **WAPE 1.2580**, $R^2$ **0.0681**.
  - **Croston (Đơn biến):** MAE 5.5567, RMSE 26.2292, WAPE 1.5439, $R^2$ 0.0311.
  - **SBA (Đơn biến):** MAE 5.8030, RMSE 34.2119, WAPE 1.6123, $R^2$ -0.6484.
  - **Baseline Lag-1:** MAE 5.2342, RMSE 34.5910, WAPE 1.4543, $R^2$ -0.6852.
- **Nhận định:** TSB vượt trội so với Croston/SBA truyền thống nhờ cơ chế cập nhật suy giảm xác suất bán $p_t$ liên tục ngay cả trong những chuỗi ngày không phát sinh đơn hàng ($y_t=0$), nhưng vẫn đứng sau mô hình đa biến `TwoStageDemand` do thiếu thông tin lịch/giá/nhóm sản phẩm.

### 2. Tổng hợp dự báo nhu cầu theo tuần (Weekly Aggregation)
- Tập dữ liệu tuần chuẩn hóa (chỉ giữ đúng 3.271 sản phẩm có $\ge 10$ ngày bán trong train, các tuần đủ 7 ngày và không vắt qua ranh giới train/val/test): 340.184 dòng (`train`: 166.821, `val`: 39.252, `test`: 127.569), tỷ lệ tuần bằng 0 giảm xuống **49.11%** (53.26% trên test).
- **Kết quả tập Test theo tuần (127.569 quan sát):**
  - **HistGB_L1 (Được chọn theo Val WAPE 0.7932):** MAE **16.3357**, RMSE 71.1005, **WAPE 0.6565**, $R^2$ **0.4302**.
  - **TwoStageWeekly (Hurdle tuần):** MAE 17.0064, RMSE 69.7045, **WAPE 0.6835**, $R^2$ **0.4524**.
  - **Daily_TwoStage_SummedToWeekly (Cộng gộp dự báo ngày lên tuần):** MAE 17.8472, RMSE **68.6813**, WAPE 0.7173, $R^2$ **0.4684**.
  - **Lag1W_Baseline:** MAE 21.3067, RMSE 87.7804, WAPE 0.8563, $R^2$ 0.1316.
  - **Ridge / XGBoost / HistGB_L2 / RandomForest (tối ưu L2):** WAPE từ 0.8962 đến 0.9939, $R^2$ từ 0.1905 đến 0.4246.
- **Nhận định:** Sau khi loại bỏ rò rỉ tuần biên (boundary-week leakage), loại bỏ `.bfill()` giá và đồng bộ tập 3.271 sản phẩm, dự báo trực tiếp theo tuần bằng hàm mất mát L1 (`HistGB_L1` và `TwoStageWeekly`) kéo WAPE xuống **0.6565–0.6835** và nâng $R^2$ lên **0.43–0.45**, vượt qua cả phương pháp cộng gộp dự báo ngày (`0.7173`) lẫn baseline tuần (`0.8563`).

### 3. Đánh giá chi phí tồn kho thực tế (Inventory Cost & Newsvendor)
- **Phát hiện QA quan trọng (Zero-Inflation Quantile Collapse):** Vì 78.9% dòng train và 81.85% dòng test có `DailyDemand = 0`, phân vị vô điều kiện $\tau = 0.75$ gần như bằng 0 trên toàn bộ mẫu. Do đó `Quantile_0.75` vô điều kiện dự đoán xấp xỉ 0 (Volume Fill Rate chỉ đạt **0.17%**, Positive-Day Service Level **0.02%**), dẫn đến tổng chi phí thiếu hàng rất lớn ($10.02M ở kịch bản $C_u=3, C_o=1$).
- **Giải pháp `Hurdle_Newsvendor_Opt`:** Tách xác suất bán $P(Y>0|X)$ và hồi quy phân vị có điều kiện trên miền dương ($Y>0$), sau đó tối ưu ngưỡng quyết định trực tiếp theo hàm chi phí $(C_u, C_o)$ trên tập validation:
  1. *Cân bằng ($C_u=1, C_o=1$)*: **Hurdle_Newsvendor_Opt** đạt chi phí thấp nhất (**$3.02M** vs TwoStageDemand **$3.61M**, TSB **$4.21M**, Lag-1 **$4.86M**).
  2. *Bán lẻ chuẩn ($C_u=3, C_o=1$)*: **Hurdle_Newsvendor_Opt** đạt chi phí thấp nhất (**$7.61M**, Volume Fill Rate 36.69%) so với TwoStageDemand (**$7.75M**), TSB (**$8.43M**), Quantile_0.90 (**$9.14M**, Volume Fill Rate **62.91%**), Lag-1 (**$9.73M**) và Quantile_0.75 (**$10.02M**).
  3. *Khách hàng trung thành / VIP ($C_u=9, C_o=1$)*: **Quantile_0.90** (**$16.58M**, Volume Fill Rate **62.91%**, Positive-Day SL **71.60%**) và **Hurdle_Newsvendor_Opt** (**$17.14M**, Volume Fill Rate **58.37%**) giảm mạnh chi phí so với TwoStageDemand (**$20.17M**), TSB (**$21.12M**) và Lag-1 (**$24.31M**).
  4. *Lưu kho đắt đỏ ($C_u=1, C_o=2$)*: **Hurdle_Newsvendor_Opt** đạt chi phí thấp nhất (**$3.18M**).
