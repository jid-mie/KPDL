# Checklist điều kiện dự án (đối chiếu KPDL.md)

| Điều kiện | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| Nạp cả hai sheet Online Retail II, chuẩn hóa schema | Đạt | `preprocess.py`; báo cáo 1.067.371 dòng nguồn |
| Loại hóa đơn hủy, quantity âm, price không dương, duplicate | Đạt theo quy tắc đề cương | `reports/preprocessing_report.json`; 59.458 bản ghi bị loại theo quality rules |
| Xử lý Description thiếu, giữ CustomerID thiếu không điền giả | Đạt | Missing CustomerID 228.488 sau làm sạch; không dùng làm feature dự báo |
| Loại phí/dịch vụ khỏi nhánh nhu cầu | Đạt | 4.260 dòng bị loại khỏi demand aggregation, vẫn giữ trong clean transactions |
| Tổng hợp product/day và điền 0 cho ngày không bán | Đạt | 3.497.687 dòng, không trùng khóa (StockCode, Date), có 2.967.589 ngày-sản-phẩm demand 0 |
| Đặc trưng lịch sử chỉ nhìn quá khứ | Đạt | Lag/rolling shift tối thiểu 1; invoice count và giá trong model đều dùng lag |
| Đặc trưng sản phẩm/K-Means chỉ fit train | Đạt | `ProductCluster`, sản lượng/tần suất/giá trung bình tính từ train |
| Cờ ngày lễ | Đạt | Lịch UK qua `holidays`; cần nêu rõ vì dữ liệu chủ yếu UK nhưng gồm 43 quốc gia |
| EDA theo tuần/tháng, sản phẩm, khách hàng/quốc gia | Đạt | Có customer/country summary, phân vị chi tiêu khách hàng, top khách/quốc gia và biểu đồ trong `reports/` |
| K-Means, Elbow, Silhouette | Đạt | k=3, silhouette=0,5464; cluster report có hồ sơ cụm |
| Apriori theo hóa đơn | Đạt | 5.187 luật đầy đủ ở JSON/CSV; có nhóm luật xếp hạng và diễn giải trong `association_interpretation.md` |
| Bất thường tổng Quantity và Revenue theo ngày | Đạt | 30 cờ Quantity, 29 cờ Revenue bằng trailing median 28 ngày |
| Bất thường nhu cầu theo product/day | Đạt về triển khai; ngưỡng cần nghiệp vụ duyệt | Không xét ngày không bán; 41.168/530.098 ngày bán (7,766%) bị gắn cờ với cửa sổ 90 ngày, Q3+1,5×IQR; không tự loại khỏi dữ liệu |
| Chia Train/Validation/Test theo ngày trong tài liệu | Đạt | Train đến 2010-11-30; val đến 2011-02-28; test đến 2011-12-09; không giao nhau |
| So sánh Linear Regression, Random Forest, XGBoost | Đạt | Đã chạy cả ba; thêm HistGradientBoosting làm đối chứng |
| Chọn mô hình bằng validation, đánh giá test riêng | Đạt | HistGradientBoosting được chọn theo MAE validation |
| Chất lượng dự báo phù hợp dùng thực tế | Chưa đạt | TwoStageDemand cải thiện test: MAE 3,881; RMSE 24,793; WAPE 1,078; R² 0,134; tốt hơn Lag-1 nhưng WAPE vẫn >1 |
| Không leakage từ target hoặc dữ liệu cùng ngày | Đạt sau kiểm tra/sửa | Predictor không có biến `DailyDemand*`, `UnitPrice` hoặc `InvoiceCount` cùng ngày; giá/invoice dùng lag |
| Có báo cáo/biểu đồ đầu ra | Đạt một phần | JSON/CSV và biểu đồ pipeline có; cần bổ sung phần diễn giải kết quả, phương pháp, hạn chế thành báo cáo nộp hoàn chỉnh |

## Kết luận

Pipeline kỹ thuật chính đã chạy xuyên suốt, các mốc chia và điều kiện tránh leakage quan trọng đã được kiểm tra. Bốn hạng mục tồn đọng đã được xử lý trong mã/báo cáo. Tuy vậy chất lượng dự báo và ngưỡng anomaly vẫn cần xác nhận theo mục tiêu nghiệp vụ trước khi dùng vận hành.
