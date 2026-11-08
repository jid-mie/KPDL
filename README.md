# KPDL — Online Retail II

Pipeline khai phá dữ liệu và dự báo nhu cầu sản phẩm theo tài liệu `KPDL.md`.

## Cài đặt và chạy

```bash
uv sync
uv run python preprocess.py
uv run python eda.py
uv run python feature_engineering.py
uv run python mining.py
uv run python modeling.py
```

`clustering.py` có thể chạy riêng để xuất báo cáo K-Means độc lập; bước feature engineering cũng tự tạo nhãn cụm từ phần train để bộ feature luôn nhất quán.

## Đầu ra

- `data/processed/`: giao dịch đã làm sạch, nhu cầu theo ngày, cụm sản phẩm.
- `data/model/`: feature matrix và các tập train/validation/test; dự đoán test nằm trong `data/model/predictions/`.
- `reports/`: báo cáo JSON/CSV về tiền xử lý, EDA, luật kết hợp, bất thường, phân cụm và đánh giá mô hình; biểu đồ trong `reports/figures/`.

Các đặc trưng biến đổi trực tiếp từ `DailyDemand` chỉ phục vụ phân tích, không được dùng làm đầu vào mô hình. Chọn mô hình bằng validation; test được giữ cho đánh giá cuối.

## Báo cáo phân tích mở rộng

- EDA khách hàng/quốc gia: `reports/customer_summary.csv`, `reports/country_summary.csv`, `reports/figures/customer_revenue_distribution.png`, `reports/figures/country_revenue_top10.png`.
- Apriori đầy đủ: `reports/association_rules.csv` và `.json`; phần diễn giải: `reports/association_interpretation.md`.
- Bất thường product/day: `reports/product_demand_anomalies.parquet`; cờ anomaly chỉ xét ngày có bán, dùng Q3 + 1,5×IQR trên cửa sổ lịch sử 90 ngày. Các cờ cần được xem xét, không tự động đồng nghĩa lỗi.
- Dự báo hurdle/two-stage: model lựa chọn cutoff bán từ validation, kết quả cuối ở `reports/modeling_report.json`.

## Các phân hệ nâng cấp mới (Intermittent, Weekly & Inventory Optimization)

1. **Mô hình Croston, SBA & TSB (`intermittent_models.py`):**
   - Chuyên biệt cho chuỗi thời gian nhu cầu gián đoạn (78–82% số 0) trên đúng 3.271 mã sản phẩm chuẩn.
   - TSB ($\alpha=0.20, \beta=0.40$) đạt WAPE **1.2580**, RMSE **25.72**, $R^2$ **0.0681** (vượt trội so với Croston 1.5439, SBA 1.6123 và Baseline Lag-1 1.4543; đứng sau mô hình đa biến TwoStageDemand 1.0782).
   - Báo cáo: `reports/intermittent_models_report.json`, dự đoán test: `data/model/predictions/intermittent_test_predictions.parquet`, biểu đồ: `reports/figures/intermittent_models_comparison.png`.

2. **Tổng hợp dự báo theo tuần (`weekly_forecasting.py`):**
   - Gom nhóm theo chu kỳ tuần 7 ngày đầy đủ (`WeeklyDemand`) trên cùng tập 3.271 sản phẩm, không chồng lấn ranh giới train/val/test, giảm tỷ lệ số 0 từ ~80% xuống **49.11%** (53.26% trên test).
   - **HistGB_L1** đạt WAPE **0.6565**, $R^2$ **0.4302** và **TwoStageWeekly** đạt WAPE **0.6835**, $R^2$ **0.4524** (tốt hơn cả tổng hợp dự báo ngày `Daily_TwoStage_SummedToWeekly` có WAPE 0.7173 và `Lag1W_Baseline` có WAPE 0.8563).
   - Báo cáo: `reports/weekly_modeling_report.json`, biểu đồ: `reports/figures/weekly_model_comparison.png`.

3. **Tối ưu chi phí tồn kho thực tế (`inventory_cost.py`):**
   - Đánh giá theo hàm Newsvendor phi đối xứng giữa chi phí thiếu hàng ($C_u$) và chi phí lưu kho ($C_o$), tách bạch `service_level_pct` toàn cục với `positive_day_service_level_pct` và `volume_fill_rate_pct`.
   - Chỉ rõ hiện tượng **Zero-Inflation Quantile Collapse** ở `Quantile_0.75` vô điều kiện (Volume Fill Rate chỉ đạt 0.17% do ~80% dữ liệu bằng 0) và khắc phục bằng **Hurdle_Newsvendor_Opt** (kết hợp xác suất bán với phân vị có điều kiện $Y > 0$), đạt chi phí thấp nhất ở kịch bản Bán lẻ chuẩn ($C_u=3, C_o=1$: **$7.61M**) và cùng **Quantile_0.90** (**$16.58M**, Volume Fill Rate **62.91%**) giảm mạnh chi phí ở kịch bản VIP ($C_u=9, C_o=1$).
   - Báo cáo: `reports/inventory_cost_report.json`, biểu đồ: `reports/figures/inventory_cost_scenarios.png` và `reports/figures/service_level_tradeoff.png`.

```bash
# Chạy 3 module nâng cấp
uv run python intermittent_models.py
uv run python weekly_forecasting.py
uv run python inventory_cost.py
```
