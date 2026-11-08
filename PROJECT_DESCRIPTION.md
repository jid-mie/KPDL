# KPDL — Tài liệu mô tả dự án

**Tên đề tài:** Khai phá dữ liệu bán lẻ Online Retail II và dự báo nhu cầu sản phẩm theo ngày

**Bộ dữ liệu:** Online Retail II (UCI Machine Learning Repository), giai đoạn 2009-12-01 → 2011-12-09

**Môi trường:** Python ≥ 3.14, quản lý bằng `uv`; thư viện: pandas 3, scikit-learn 1.9, xgboost 3.1, pyarrow, matplotlib, holidays, openpyxl

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Mục tiêu và phạm vi](#2-mục-tiêu-và-phạm-vi)
3. [Dữ liệu](#3-dữ-liệu)
4. [Kiến trúc pipeline](#4-kiến-trúc-pipeline)
5. [Tiền xử lý](#5-tiền-xử-lý)
6. [Phân tích dữ liệu (EDA)](#6-phân-tích-dữ-liệu-eda)
7. [Khai phá dữ liệu](#7-khai-phá-dữ-liệu)
8. [Feature engineering](#8-feature-engineering)
9. [Chia tập dữ liệu](#9-chia-tập-dữ-liệu)
10. [Mô hình và đánh giá](#10-mô-hình-và-đánh-giá)
11. [Kết quả thực nghiệm](#11-kết-quả-thực-nghiệm)
12. [Đánh giá mức độ đáp ứng đề cương](#12-đánh-giá-mức-độ-đáp-ứng-đề-cương)
13. [Hạn chế và hướng phát triển](#13-hạn-chế-và-hướng-phát-triển)
14. [Cấu trúc thư mục và cách chạy](#14-cấu-trúc-thư-mục-và-cách-chạy)
15. [Thuật toán chi tiết](#15-thuật-toán-chi-tiết)
16. [Chỉ số và siêu tham số](#16-chỉ-số-và-siêu-tham-số)
17. [Phụ lục: định nghĩa chỉ số](#17-phụ-lục-định-nghĩa-chỉ-số)

---

## 1. Tổng quan

Dự án xây dựng một quy trình khai phá dữ liệu hoàn chỉnh trên dữ liệu giao dịch bán lẻ, nhằm trả lời hai nhóm câu hỏi:

**Nhóm mô tả (descriptive):**
- Cấu trúc doanh thu, số lượng bán và khách hàng của tập dữ liệu như thế nào?
- Sản phẩm được chia thành những nhóm nào theo quy mô và mức độ biến động nhu cầu?
- Những sản phẩm nào thường được mua cùng nhau?
- Những ngày nào có nhu cầu/doanh thu bất thường?

**Nhóm dự báo (predictive):**
- Có thể dự đoán nhu cầu của từng sản phẩm trong ngày kế tiếp không?
- Mô hình nào phù hợp nhất với chuỗi nhu cầu gián đoạn (intermittent demand)?
- Sai số dự báo có đủ nhỏ để dùng cho lập kế hoạch tồn kho không?

Bản chất bài toán dự báo: **hồi quy chuỗi thời gian đa chuỗi** (one model for all products), biến đích là `DailyDemand = Σ Quantity` của một `StockCode` trong một ngày, dự báo **1 ngày tới** dựa trên lịch sử trước đó.

---

## 2. Mục tiêu và phạm vi

### 2.1. Mục tiêu

| # | Mục tiêu | Tiêu chí hoàn thành |
|---|---|---|
| M1 | Làm sạch dữ liệu giao dịch, kiểm soát chất lượng có thể kiểm chứng | Mỗi bước lọc ghi lại số dòng bị loại, tái lập được |
| M2 | Tổng hợp nhu cầu sản phẩm/ngày, biểu diễn ngày không bán bằng 0 | Bảng đầy đủ `(StockCode, Date)`, không trùng khóa |
| M3 | Mô tả dữ liệu theo 4 hướng: thống kê, thời gian, sản phẩm, khách hàng/quốc gia | Bảng CSV + biểu đồ + số liệu cụ thể |
| M4 | Phân cụm sản phẩm bằng K-Means, chọn k bằng Elbow + Silhouette | Có báo cáo cụm và hồ sơ từng cụm |
| M5 | Khai phá luật kết hợp bằng Apriori trên từng hóa đơn | Luật đầy đủ (support, confidence, lift) |
| M6 | Phát hiện bất thường theo ngày (tổng lượng, doanh thu) | Danh sách ngày bị gắn cờ + phương pháp |
| M7 | Xây dựng đặc trưng lịch sử/giá/sản phẩm **không rò rỉ tương lai** | Lag/rolling dùng `shift(≥1)`; thống kê sản phẩm chỉ từ train |
| M8 | So sánh ≥ 3 mô hình, chọn bằng validation, đánh giá test một lần | Bảng metric validation + test + baseline |
| M9 | Trực quan hóa kết quả | Biểu đồ trong `reports/figures/` |

### 2.2. Ngoài phạm vi

- Không dự báo nhiều ngày (multi-step horizon > 1).
- Không phân loại/segment khách hàng (RFM, churn).
- Không gợi ý sản phẩm theo người dùng; Apriori chỉ ở mức giỏ hàng.
- Không tối ưu tồn kho bằng hàm chi phí; không có quyết định tự động.

---

## 3. Dữ liệu

### 3.1. Nguồn

`online_retail_II.xlsx` (43,5 MB), 2 sheet: `Year 2009-2010`, `Year 2010-2011`.

| Thuộc tính | Ý nghĩa | Vai trò trong dự án |
|---|---|---|
| `InvoiceNo` | Mã hóa đơn | Xác định giao dịch; tiền tố `C` = hủy |
| `StockCode` | Mã sản phẩm | Khóa của chuỗi nhu cầu |
| `Description` | Tên sản phẩm | EDA; điền thiếu theo mode |
| `Quantity` | Số lượng | Nguồn của `DailyDemand` |
| `InvoiceDate` | Thời điểm giao dịch | Tách ngày, tạo đặc trưng thời gian |
| `UnitPrice` | Đơn giá | Đặc trưng giá (dạng lag) |
| `CustomerID` | Mã khách hàng | EDA khách hàng; **không** dùng cho model |
| `Country` | Quốc gia | EDA khu vực; không dùng cho model |

### 3.2. Chất lượng dữ liệu thô

- 1.067.371 dòng nguồn.
- `CustomerID` thiếu 243.007 (22,77%).
- `Description` thiếu 4.382 (0,41%).
- Có hóa đơn hủy (`C…`), số lượng âm, đơn giá ≤ 0, dòng trùng lặp hoàn toàn.
- `StockCode` chứa mã phí/dịch vụ không có chữ số (`POST`, `BANK CHARGES`, `DOT`, …).

---

## 4. Kiến trúc pipeline

```
online_retail_II.xlsx
        │
        ▼
  preprocess.py ──────────► data/processed/transactions_clean.parquet
        │                   data/processed/product_daily_demand.parquet
        │                   reports/preprocessing_report.json
        ├──► eda.py ───────► reports/eda_report.json + monthly/weekly/product/
        │                    customer/country CSV + 5 biểu đồ
        │
        ▼
 feature_engineering.py ──► data/model/{train,val,test,features_full}.parquet
        │                   reports/feature_engineering_report.json
        ├──► clustering.py ► data/processed/product_clusters.{parquet,csv}
        │                   reports/clustering_report.json
        ├──► mining.py ────► reports/association_rules.{csv,json}
        │                   reports/association_interpretation.md
        │                   reports/daily_anomalies.csv, anomaly_report.json
        │                   reports/product_demand_anomalies.parquet
        │
        ▼
    modeling.py ─────────► reports/modeling_report.json
                           data/model/predictions/test_predictions.parquet
                           reports/figures/demand_forecast_test.png
```

Đặc điểm thiết kế:

- **Mỗi script độc lập, đọc/ghi file trung gian** → có thể chạy lại riêng từng bước.
- **Không có state ẩn**: mọi ngưỡng là hằng số ở đầu file.
- **Báo cáo có thể kiểm toán**: mọi bước lọc đều ghi số dòng bị loại và tỷ lệ.
- **Không dùng notebook** → tái lập bằng CLI.

---

## 5. Tiền xử lý

File: `preprocess.py` (204 dòng). Output: `data/processed/*`, `reports/preprocessing_report.json`.

### 5.1. Chuẩn hóa schema

Tên cột khác nhau giữa hai sheet (`Invoice` vs `InvoiceNo`, `Price` vs `UnitPrice`, `Customer ID` vs `CustomerID`) được ánh xạ qua `find_column()` — so khớp sau khi bỏ khoảng trắng và hạ chữ thường. Thêm cột `SourceSheet` để truy vết nguồn.

Chuẩn hóa giá trị:

- `InvoiceNo`, `StockCode`, `Country`: strip, `StockCode` viết hoa.
- `Description`: gộp khoảng trắng thừa, strip, viết hoa.
- `Quantity`, `UnitPrice`: ép số, lỗi → NaN.
- `InvoiceDate`: `pd.to_datetime(..., dayfirst=True)`.
- `CustomerID`: `Int64` nullable (không điền).

### 5.2. Loại bản ghi nhiễu (thứ tự cố định, loại trừ lẫn nhau)

| Bước | Quy tắc | Số dòng | % nguồn |
|---|---|---:|---:|
| `cancelled_invoice` | `InvoiceNo` bắt đầu bằng `C` | 19.494 | 1,8264 |
| `negative_quantity` | `Quantity < 0` còn lại | 3.457 | 0,3239 |
| `nonpositive_price` | `UnitPrice ≤ 0` | 2.750 | 0,2576 |
| `exact_duplicate` | trùng toàn bộ 8 cột, giữ bản đầu | 33.757 | 3,1626 |
| `unresolved_product_description_or_code` | không điền được Description và không có StockCode | 0 | 0 |
| `missing_or_invalid_core_field` | NaN ở InvoiceNo/Quantity/UnitPrice/InvoiceDate/Country | 0 | 0 |
| **Tổng** | | **59.458** | **5,57** |

Thứ tự quan trọng: lọc hủy trước, rồi quantity âm, rồi giá, rồi trùng lặp — để bảng báo cáo không bị đếm trùng dòng.

### 5.3. Xử lý thiếu

- `Description`: điền bằng **mode** của các Description cùng `StockCode` (đã chuẩn hóa). Không tìm được → loại dòng.
- `CustomerID`: **không điền**. Giữ nullable vì không phải biến dự báo; điền giá trị giả sẽ tạo cảm giác dữ liệu đầy đủ.
- `Quantity`, `UnitPrice`, `InvoiceDate`, `Country`: sau lọc không còn NaN.

Sau làm sạch: **1.007.913 dòng**, 4.745 mã hàng, 5.878 khách hàng có ID, 43 quốc gia, `CustomerID` thiếu 228.488.

### 5.4. Tách nhánh dữ liệu

- `transactions_clean`: toàn bộ dòng sạch + `Revenue = Quantity × UnitPrice` + `Date`/`Year`/`Month`/`Day`/`DayOfWeek`/`WeekOfYear`. Dùng cho EDA, clustering, Apriori, anomaly.
- Nhánh nhu cầu: chỉ giữ `StockCode` **có chữ số** (`str.contains(r'\d')`), loại 4.260 dòng phí/dịch vụ.

### 5.5. Tổng hợp nhu cầu

```
DailyDemand  = Σ Quantity          (theo StockCode, Date)
UnitPrice    = mean(UnitPrice)     (theo StockCode, Date)
InvoiceCount = nunique(InvoiceNo)  (theo StockCode, Date)
```

Sau đó **reindex đầy đủ** tích Descartes `(StockCode × mọi ngày trong khoảng dữ liệu)`:

- `DailyDemand = 0` cho ngày không bán (không phải NaN) — đây là quyết định quan trọng nhất của bước này, tránh chuỗi thời gian bị lệch.
- `InvoiceCount = 0`.
- `UnitPrice` **giữ NaN** trên ngày không bán: giá không xác định, không bịa quan sát.

Kết quả: **3.497.687 dòng** `(StockCode, Date)`, trong đó 2.967.589 dòng demand = 0 (84,8%).

---

## 6. Phân tích dữ liệu (EDA)

File: `eda.py` (45 dòng). Output: `reports/eda_report.json`, 5 CSV, 5 biểu đồ.

### 6.1. Thống kê mô tả

| Chỉ số | Giá trị |
|---|---|
| Dòng sạch | 1.007.913 |
| Sản phẩm | 4.745 |
| Khách hàng có ID | 5.878 |
| Quốc gia | 43 |
| Khoảng thời gian | 2009-12-01 → 2011-12-09 |
| Quantity: mean / median / P90 / P95 / P99 / max | 11,12 / 4 / 24 / 31 / 108 / 80.995 |
| UnitPrice: mean / median / P95 / max | 4,07 / 2,10 / 9,95 / 25.111,09 |

Phân phối Quantity lệch phải rất mạnh (mean 11,12 ≫ median 4) → củng cố quyết định dùng mô hình hai giai đoạn và biến đổi log/sqrt.

### 6.2. Theo thời gian

- Doanh thu và số lượng theo **tuần** (`W-MON`) và **tháng** (`MS`): 25 tháng, 106 tuần.
- Đỉnh: **2011-11** với 751.377 đơn vị và 1.503.866,78 doanh thu → xác nhận mùa vụ cuối năm.
- Biểu đồ: `monthly_quantity.png`, `monthly_revenue.png`, `weekly_quantity.png`.

### 6.3. Theo sản phẩm

- `product_summary.csv`: số hóa đơn, tổng lượng, doanh thu theo `StockCode`.
- **10% sản phẩm đứng đầu chiếm 63,17% doanh thu** → phân bố tập trung cao (Pareto).
- Top 10 sản phẩm theo doanh thu lưu trong `eda_report.json`.

### 6.4. Theo khách hàng và quốc gia

- `customer_summary.csv`: số hóa đơn, lượng, doanh thu, lần mua đầu/cuối.
- Phân vị doanh thu/khách: P25 = 342,28; median = 867,74; P75 = 2.248,31; P90 = 5.465,74; P99 = 29.205,90; max = 580.987,04.
- `country_summary.csv`: 3 quốc gia đầu chiếm **90,95%** doanh thu (Anh chiếm áp đảo).
- Biểu đồ: `customer_revenue_distribution.png` (clip P99), `country_revenue_top10.png`.

### 6.5. Nhận xét từ EDA

1. Nhu cầu gián đoạn: 84,8% quan sát product/ngày bằng 0 → cần mô hình xử lý zero-inflation, không nên hồi quy thuần.
2. Tập trung doanh thu: 10% sản phẩm = 63% doanh thu → sai số ở nhóm chủ lực ảnh hưởng lớn.
3. Mùa vụ rõ rệt, đỉnh tháng 11 → chia theo thời gian là bắt buộc.
4. `CustomerID` thiếu 22,7% và lệch tập trung → loại khỏi biến dự báo là hợp lý.

---

## 7. Khai phá dữ liệu

### 7.1. Phân cụm sản phẩm — K-Means

File: `clustering.py` (182 dòng). Output: `data/processed/product_clusters.{parquet,csv}`, `reports/clustering_report.json`.

**Không gian đặc trưng** (tính **chỉ trên train**, `Date ≤ 2010-11-30`):

| Đặc trưng | Ý nghĩa |
|---|---|
| `AvgDailyDemand` | Trung bình lượng bán/ngày |
| `ActiveDays` | Số ngày phát sinh bán (tần suất) |
| `AvgUnitPrice` | Giá trung bình |
| `DemandStd` | Độ biến động nhu cầu |

Chuẩn hóa `StandardScaler` trước khi fit. Đánh giá k = 2..8 bằng inertia (Elbow) và silhouette; `n_init=10`, `random_state=42`.

**Kết quả chọn k = 3** (diễn giải nghiệp vụ; xem ghi chú bên dưới):

| Cụm | Tên | Sản phẩm | % | AvgDailyDemand | ActiveDays | AvgUnitPrice | DemandStd |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Phổ thông (nhu cầu thấp–vừa) | 2.548 | 77,9% | 1,73 | 48,8 | 4,36 | 9,82 |
| 1 | Bán chạy (trung bình–cao, ổn định) | 676 | 20,67% | 11,88 | 179,9 | 3,03 | 33,49 |
| 2 | Chủ lực (đột biến, số lượng lớn) | 47 | 1,44% | 58,19 | 129,2 | 1,25 | 359,90 |

Silhouette theo k: 2 → 0,6250; **3 → 0,5464**; 4 → 0,5557; 5 → 0,5319; 6 → 0,4318; 7 → 0,4331; 8 → 0,4620.

> **Ghi chú trung thực:** k = 2 có silhouette cao nhất nhưng tạo hai nhóm quá thô; k = 3 được chọn vì ranh giới Phổ thông / Bán chạy / Chủ lực có ý nghĩa vận hành và Elbow giảm mạnh từ k=2 (9.386) xuống k=3 (6.969). Đây là lựa chọn có chủ đích, không phải kết quả thuần túy của silhouette — cần nêu rõ khi bảo vệ.

**Vai trò trong pipeline:** nhãn cụm được tái tạo trong `feature_engineering.py` (fit lại trên train, `k=3`) và đưa vào mô hình dưới dạng `ProductCluster` (one-hot), bảo đảm feature luôn nhất quán và không rò rỉ.

### 7.2. Luật kết hợp — Apriori

File: `mining.py` (164 dòng). Output: `reports/association_rules.{csv,json}`, `reports/association_interpretation.md`.

Cài đặt Apriori thủ công, không dùng `mlxtend`:

- Mỗi `InvoiceNo` = một basket chứa các `StockCode` có chữ số.
- Loại hóa đơn hủy.
- `min_support = 0,005`, `min_confidence = 0,15`, `max_itemset_length = 2`.
- Tối ưu: với k=2 chỉ sinh cặp từ các item phổ biến trong basket thay vì quét mọi candidate.

**Kết quả:** 39.536 hóa đơn → **5.187 luật**; 932 luật có support ≥ 1% được ưu tiên diễn giải.

Luật mạnh nhất theo lift (nhóm mã sản phẩm liên quan, lift ≈ 46):

| Nếu có | Thì có | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| 22746 | 22745 | 1,17% | 81,74% | 45,97 |
| 22745 | 22746 | 1,17% | 65,58% | 45,97 |
| 22746 | 22748 | 1,20% | 84,04% | 44,78 |
| 22748 | 22746 | 1,20% | 63,88% | 44,78 |

Diễn giải trong `association_interpretation.md` nêu rõ: support thấp + lift cao có thể do sản phẩm hiếm; đồng mua **không** chứng minh nhân quả.

### 7.3. Phát hiện bất thường

File: `mining.py::detect_anomalies`. Output: `reports/daily_anomalies.csv`, `reports/anomaly_report.json`, `reports/product_demand_anomalies.parquet`.

Ba phương pháp song song:

**(a) Ngưỡng rolling — tổng lượng và doanh thu theo ngày**

```
baseline_t = median(series[t-28 .. t-1])
cờ bất thường nếu series_t / baseline_t ≥ 3 hoặc ≤ 0,33
```

Yêu cầu tối thiểu 7 quan sát lịch sử. Kết quả: 604 ngày, **30 ngày bất thường theo Quantity**, **29 theo Revenue**. Top ngày: 2010-09-27 (125.489 đơn vị), 2010-08-09 (100.224), 2010-02-15 (96.425), 2011-12-09 (93.950), 2011-01-18 (82.964).

**(b) Isolation Forest — tổng hợp**

Fit trên `log1p` của `TotalQuantity`, `TotalRevenue`, `InvoiceCount`; `contamination = 0,02`, `random_state=42` → **13 ngày** bị gắn cờ. Chỉ mang tính khám phá.

**(c) Tukey fence theo sản phẩm/ngày (mở rộng ngoài đề cương)**

- Chỉ xét ngày **có bán** (tránh gắn cờ ngày 0).
- Cửa sổ 90 quan sát lịch trước đó, tối thiểu 12 quan sát dương.
- Ngưỡng `Q3 + 1,5 × IQR` của chính sản phẩm.

Kết quả: **41.168 / 530.098 ngày bán (7,766%)** bị gắn cờ. Cờ **không** tự động xóa dòng; đây là danh sách cần chuyên gia nghiệp vụ rà soát.

---

## 8. Feature engineering

File: `feature_engineering.py` (371 dòng). Output: `data/model/*.parquet`, `reports/feature_engineering_report.json`.

### 8.1. Các bước

| Bước | Nội dung |
|---|---|
| 1 | Cờ ngày lễ UK qua thư viện `holidays` → `IsHoliday` |
| 2 | Lọc sản phẩm ít giao dịch: giữ sản phẩm có ≥ **10 ngày bán trong train** |
| 3 | Gắn cờ outlier theo sản phẩm (IQR trên demand dương) — **chỉ để phân tích, không dùng làm feature** |
| 4 | Đặc trưng lịch sử: lag, rolling, recency |
| 5 | Đặc trưng giá: `PriceLag_1`, `PriceChange`, `InvoiceCountLag_1` |
| 6 | Biến đổi target: log1p, sqrt, phân lớp — **chỉ để phân tích** |
| 7 | Đặc trưng sản phẩm + nhãn cụm, tính **chỉ từ train** |
| 8 | Chia train/val/test theo thời gian |

Lọc sản phẩm: 4.733 → **3.271 sản phẩm** (loại 1.462); 3.497.687 → **2.417.269 dòng**.

Gắn cờ outlier: 38.533 dòng (1,594%) — giữ nguyên trong dữ liệu.

### 8.2. Đặc trưng mô hình (23 biến)

| Nhóm | Biến |
|---|---|
| Thời gian (6) | `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear`, `IsHoliday` |
| Lag (4) | `Lag_1`, `Lag_7`, `Lag_14`, `Lag_28` |
| Rolling (5) | `RollingMean_7`, `RollingMean_14`, `RollingMean_28`, `RollingSum_7`, `RollingStd_28` |
| Recency (1) | `DaysSinceLastSale` |
| Giá (3) | `PriceLag_1`, `PriceChange`, `AvgUnitPrice` |
| Sản phẩm (3) | `TotalActiveDays`, `TotalQuantitySold`, `ProductCluster` |
| Hóa đơn (1) | `InvoiceCountLag_1` |

### 8.3. Kiểm soát rò rỉ dữ liệu

Đây là phần dễ sai nhất của bài toán chuỗi thời gian, được xử lý bằng 4 quy tắc:

1. **Mọi lag/rolling dùng `shift(1)` trước khi tính** — rolling mean/std trên chuỗi đã shift, không dùng giá trị ngày hiện tại.
2. **`DaysSinceLastSale`** lấy ngày bán gần nhất đã `shift(1)`, không tính ngày đang dự đoán.
3. **Giá**: `UnitPrice` được forward-fill qua ngày không bán **rồi mới shift 1**; `InvoiceCount` cũng dùng bản lag.
4. **Thống kê sản phẩm và K-Means** chỉ fit trên train rồi map sang val/test.

Các cột bị loại khỏi predictor: `DailyDemand*` (kể cả log/sqrt/category/is_high), `UnitPrice` cùng ngày, `InvoiceCount` cùng ngày, `IsOutlierDemand`, `ProductClusterName`, `Date`, `StockCode`.

> Rủi ro còn lại: `ProductCluster` được chuyển bằng one-hot nhưng `TotalQuantitySold`/`TotalActiveDays` là hằng số theo sản phẩm — với một số mô hình cây, chúng có thể đóng vai trò "nhận diện sản phẩm" hơn là tín hiệu nhu cầu. Cần kiểm tra ablation.

### 8.4. Phân bố target theo tập

| Tập | Dòng | mean | std | median | max |
|---|---:|---:|---:|---:|---:|
| Train | 1.193.915 | 4,6414 | 58,0091 | 0 | 19.152 |
| Validation | 294.390 | 3,0642 | 26,8550 | 0 | 3.906 |
| Test | 928.964 | 3,5992 | 26,6466 | 0 | 4.848 |

Cả ba tập đều có median = 0 và P75 = 0 → mức zero-inflation tương tự nhau (~78–80% quan sát bằng 0), thuận lợi cho việc so sánh metric.

---

## 9. Chia tập dữ liệu

Chia **theo thứ tự thời gian**, không ngẫu nhiên:

| Tập | Khoảng | Dòng | Mục đích |
|---|---|---:|---|
| Train | 2009-12-01 → 2010-11-30 | 1.193.915 | Huấn luyện |
| Validation | 2010-12-01 → 2011-02-28 | 294.390 | Chọn mô hình, chọn ngưỡng |
| Test | 2011-03-01 → 2011-12-09 | 928.964 | Đánh giá cuối, dùng **một lần** |

Lý do: biến đích là nhu cầu theo thời gian; chia ngẫu nhiên sẽ để mô hình "nhìn thấy" tương lai. Ba tập không giao nhau, không có khoảng trống.

Lưu ý: test chiếm 9 tháng liên tục (nhiều hơn validation) và bao trùm mùa cao điểm tháng 11/2011 — đây là phép thử khó hơn validation.

---

## 10. Mô hình và đánh giá

File: `modeling.py` (104 dòng). Output: `reports/modeling_report.json`, `data/model/predictions/*`, `reports/figures/demand_forecast_test.png`.

### 10.1. Tiền xử lý đầu vào

- Loại dòng thiếu `Lag_1` (ngày đầu tiên của mỗi sản phẩm).
- Nếu train > 250.000 dòng → **lấy mẫu ngẫu nhiên 250.000** (`random_state=42`) để giới hạn thời gian huấn luyện. `train_sampled = true` và `train_rows_used = 250.000` được ghi vào báo cáo.
- Pipeline: numeric → median imputation + StandardScaler; `ProductCluster` → most-frequent imputation + OneHotEncoder.

### 10.2. Các mô hình

| Mô hình | Cấu hình |
|---|---|
| **TwoStageDemand** (hurdle) | HGBClassifier (sale/no-sale) + HGBRegressor (lượng khi có bán, `loss=absolute_error`) |
| LinearRegression | mặc định |
| RandomForest | 120 cây, `min_samples_leaf=3`, `max_features=0.8` |
| HistGradientBoosting | `loss=absolute_error`, 250 vòng, 31 lá, `l2=2.0` |
| XGBoost | 250 cây, depth 7, `lr=0.05`, subsample/colsample 0.8, `reg:squarederror` |

**TwoStageDemand** là mô hình chính vì 78–80% quan sát bằng 0:

```
p_sale   = P(DailyDemand > 0 | X)
pos_hat  = E[DailyDemand | DailyDemand > 0, X]
dự báo   = 1{p_sale ≥ τ} × max(0, pos_hat)
```

Ngưỡng τ được quét trên lưới 0,10 → 0,90 bước 0,05, **chọn theo MAE trên validation** → τ = **0,55**. Mô hình cũng báo cáo kỳ vọng xác suất `p_sale × pos_hat` để so sánh.

Dự báo âm của mọi mô hình được cắt về 0.

### 10.3. Quy trình chọn mô hình

1. Fit tất cả mô hình trên train.
2. Tính MAE/RMSE/WAPE/R² trên validation.
3. Chọn mô hình có **MAE validation nhỏ nhất**.
4. Chỉ mô hình được chọn mới dự báo test; test không dùng để chọn.
5. So sánh với baseline **Lag-1** (dự báo bằng nhu cầu hôm qua).

---

## 11. Kết quả thực nghiệm

### 11.1. Validation

| Mô hình | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| **TwoStageDemand (τ=0,55)** | **2,9081** | 26,0989 | **0,9491** | 0,0555 |
| TwoStageDemand (kỳ vọng, không ngưỡng) | 2,9860 | 25,9976 | 0,9745 | 0,0628 |
| HistGradientBoosting | 3,0573 | 26,9878 | 0,9978 | −0,0099 |
| XGBoost | 5,1426 | 32,7638 | 1,6783 | −0,4885 |
| LinearRegression | 5,7385 | 26,0969 | 1,8728 | 0,0557 |
| RandomForest | 5,9235 | 29,9337 | 1,9332 | −0,2424 |

### 11.2. Test (đánh giá một lần)

| Mô hình | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| **TwoStageDemand** | **3,8807** | **24,7927** | **1,0782** | **0,1343** |
| Baseline Lag-1 | 5,2342 | 34,5910 | 1,4543 | −0,6852 |

Mô hình giảm MAE 25,9% và RMSE 28,3% so với baseline; R² từ −0,685 lên 0,134.

### 11.3. Diễn giải

- **Tín hiệu có thật nhưng yếu:** R² test 0,134 nghĩa là mô hình chỉ giải thích ~13% biến thiên nhu cầu ngày. Với chuỗi gián đoạn, đây là mức thường gặp nhưng chưa đủ cho quyết định tự động.
- **WAPE > 1 trên test:** tổng sai số tuyệt đối (≈ 3,60 triệu đơn vị) lớn hơn tổng nhu cầu thực (≈ 3,34 triệu). Nói cách khác, nếu dùng dự báo để đặt hàng, sai lệch tổng thể còn tệ hơn "không dự báo gì".
- **Validation tốt hơn test rõ rệt** (WAPE 0,949 → 1,078): phân phối test dịch chuyển theo mùa vụ và quy mô sản phẩm; đây là dấu hiệu distribution shift, không phải lỗi code.
- **Vì sao XGBoost kém:** dữ liệu zero-inflated + đặc trưng hằng theo sản phẩm dễ khiến boosting overfit các sản phẩm lớn; TwoStageDemand xử lý cấu trúc zero tốt hơn.
- **Kết luận sử dụng:** kết quả phù hợp để minh họa quy trình và hỗ trợ phân tích, **chưa đủ cơ sở** để lập kế hoạch tồn kho tự động.

---

## 12. Đánh giá mức độ đáp ứng đề cương

| Yêu cầu trong `KPDL.md` | Trạng thái | Ghi chú |
|---|---|---|
| Khảo sát dữ liệu thô | Đạt | Missing, trùng lặp, hủy, giá ≤ 0, định dạng ngày |
| Loại thuộc tính dư thừa | Đạt | `CustomerID`, `Country`, `Description` không vào model |
| Loại hóa đơn hủy / quantity âm / price ≤ 0 / duplicate | Đạt | Có bảng đếm từng bước |
| Điền Description theo StockCode, không điền CustomerID | Đạt | `preprocess.py:108-116` |
| Loại mã phí/dịch vụ | Đạt | 4.260 dòng tách khỏi nhánh demand |
| Tổng hợp product/day + điền 0 | Đạt | 3.497.687 dòng, 84,8% bằng 0 |
| Đặc trưng thời gian + cờ ngày lễ | Đạt | 6 biến, lịch UK |
| Đặc trưng lịch sử (7/14/28, sum, std, recency) | Đạt | 10 biến |
| Đặc trưng giá + sản phẩm + cụm | Đạt | 7 biến |
| Không rò rỉ dữ liệu tương lai | Đạt | `shift(≥1)`, fit-only-train |
| EDA 4 hướng + bảng + hình | Đạt | 5 CSV + 5 hình |
| K-Means + Elbow + Silhouette | Đạt (có lưu ý) | k=3 chọn theo diễn giải; k=2 silhouette cao hơn |
| Apriori theo hóa đơn | Đạt | 5.187 luật |
| Bất thường Quantity & Revenue theo ngày | Đạt | 30 / 29 ngày |
| Chia train/val/test đúng mốc | Đạt | Đúng ngày trong đề cương |
| So sánh Linear / RF / XGBoost | Đạt | Có thêm HGB và TwoStage |
| Chọn bằng validation, test riêng | Đạt | Test chỉ dùng một lần |
| Trực quan hóa | Đạt một phần | Có hình thời gian, KH, quốc gia, dự báo; **thiếu** hình cụm, luật, anomaly |
| Chất lượng dự báo dùng được thực tế | **Chưa đạt** | WAPE 1,078; R² 0,134 |

Hai điểm cần nêu trung thực khi báo cáo:

1. `KPDL_PROGRESS.md` ghi "XGBoost không được cài" — **đã lỗi thời**; `pyproject.toml` có `xgboost>=3.1` và kết quả XGBoost đã có trong `modeling_report.json`.
2. Train bị lấy mẫu 250.000/1.193.915 dòng (20,9%) — cần ghi rõ trong báo cáo phương pháp, vì ảnh hưởng đến khả năng học các sản phẩm hiếm.

---

## 13. Hạn chế và hướng phát triển

### 13.1. Hạn chế kỹ thuật

| Hạn chế | Ảnh hưởng | Cách xử lý đề xuất |
|---|---|---|
| Chỉ dự báo 1 ngày | Không đủ cho đặt hàng theo tuần | Thêm horizon 7/14/28 ngày, direct hoặc recursive |
| Train sample 250k | Mất mẫu sản phẩm hiếm | Train full hoặc sample theo sản phẩm (stratified) |
| Chưa validate rolling-origin | Một split duy nhất có thể may mắn | TimeSeriesSplit nhiều fold theo tháng |
| Ngưỡng anomaly heuristic | 7,77% ngày bán bị cờ — quá nhiều | Hiệu chỉnh theo chi phí false positive; thử STL residual |
| Chưa có MASE/SMAPE | WAPE khó so sánh giữa sản phẩm | Bổ sung MASE chuẩn hóa theo naive mùa vụ |
| Không có hàm chi phí tồn kho | Không biết sai số nào đắt | Thiết kế cost matrix (thiếu hàng vs tồn kho) |
| Log/sqrt chưa thực nghiệm | Đề cương yêu cầu minh họa biến đổi | So sánh target gốc vs log1p vs sqrt trên cùng validation |
| Không có test tự động | Sửa code dễ làm hỏng pipeline | Thêm assert cho schema, khóa duy nhất, không NaN ở target |

### 13.2. Hướng phát triển

1. **Dự báo theo nhóm sản phẩm**: mô hình riêng cho cụm Chủ lực (47 sản phẩm, 1,44%) và cụm Phổ thông (2.548 sản phẩm) — hai nhóm có động lực nhu cầu khác nhau.
2. **Rolling-origin validation** trên 6–12 cửa sổ để có khoảng tin cậy cho metric.
3. **Mô hình xác suất**: Poisson/NegBin/Zero-Inflated, hoặc quantile regression để có khoảng dự báo thay vì điểm.
4. **Feature bổ sung**: giá khuyến mãi, khoảng cách ngày lễ, số ngày kể từ lần bán cuối theo phân vị, độ dài chuỗi zero hiện tại.
5. **Đánh giá theo chi phí**: chuyển MAE sang chi phí thiếu hàng/tồn kho, chọn ngưỡng τ theo chi phí thay vì MAE.
6. **Hoàn thiện báo cáo**: viết Chương III (phương pháp) và Chương IV (thực nghiệm, thảo luận, hạn chế) từ các JSON/CSV hiện có.

---

## 14. Cấu trúc thư mục và cách chạy

### 14.1. Cấu trúc

```
KPDL/
├── KPDL.md                       # Đề cương gốc
├── KPDL_PROGRESS.md              # Trạng thái + kết quả (một phần đã lỗi thời)
├── PROJECT_CHECKLIST.md          # Đối chiếu đề cương ↔ triển khai
├── PROJECT_DESCRIPTION.md        # Tài liệu này
├── README.md
├── pyproject.toml / uv.lock
├── online_retail_II.xlsx          # 43,5 MB
├── preprocess.py                 # 204 dòng
├── eda.py                        # 45 dòng
├── feature_engineering.py        # 371 dòng
├── clustering.py                 # 182 dòng
├── mining.py                     # 164 dòng
├── modeling.py                   # 104 dòng
├── data/
│   ├── processed/                # transactions_clean, product_daily_demand, product_clusters
│   └── model/                    # train, val, test, features_full, predictions/
└── reports/
    ├── *.json, *.csv, *.md
    └── figures/                  # 6 biểu đồ PNG
```

### 14.2. Chạy toàn bộ

```bash
uv sync
uv run python preprocess.py
uv run python eda.py
uv run python feature_engineering.py
uv run python clustering.py
uv run python mining.py
uv run python modeling.py
```

Thứ tự bắt buộc: `preprocess` → (`eda` | `feature_engineering`) → (`clustering` | `mining` | `modeling`). `clustering.py` cần `train.parquet` nên phải chạy sau `feature_engineering.py`.

### 14.3. Thời gian chạy ước lượng

| Bước | Thời gian đo được | Ghi chú |
|---|---|---|
| preprocess | 57 giây | Đọc Excel 43,5 MB là phần chậm nhất |
| eda | 4 giây | |
| feature_engineering | 16 giây | Rolling theo 3.271 sản phẩm |
| clustering | 5 giây | |
| mining | 36 giây | Apriori trên 39,5k hóa đơn |
| modeling | 1 phút 36 giây | RF + XGBoost trên 250k dòng |
| **Tổng** | **≈ 3 phút 34 giây** | Máy 8 luồng, `n_jobs=-1` |

---

## 15. Thuật toán chi tiết

### 15.1. Tiền xử lý quy tắc (`preprocess.py`)

| Thao tác | Cơ chế | Tham số |
|---|---|---|
| Chuẩn hóa schema | So khớp tên cột sau khi bỏ space + lowercase | — |
| Lọc nhiễu | Mask tuần tự, mỗi mask loại trừ mask trước | Thứ tự: hủy → âm → giá ≤0 → duplicate |
| Điền Description | `groupby(StockCode).agg(mode)` rồi `map` | Không tìm được → xóa dòng |
| Tổng hợp | `groupby([StockCode, Date]).agg(sum, mean, nunique)` | — |
| Điền ngày 0 | `MultiIndex.from_product` → `reindex` → `fillna(0)` | Tích Descartes 3.271 SP × 730 ngày |

Không có "thuật toán học" ở bước này — chỉ là quy tắc xác định. Điểm cần hiểu: **reindex** biến "không có giao dịch" thành `0` thật, không phải NaN.

### 15.2. K-Means (`clustering.py`)

**Mục tiêu:** cực tiểu hóa inertia

```
J = Σ_{k=1..K} Σ_{x ∈ C_k} ||x − μ_k||²
```

**Cơ chế:** khởi tạo k-means++ → gán điểm vào tâm gần nhất → cập nhật tâm = trung bình cụm → lặp đến hội tụ.

**Tham số dùng trong dự án:**

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `n_clusters` | 3 | Số cụm |
| `n_init` | 10 | Chạy 10 lần khởi tạo, lấy inertia nhỏ nhất → giảm phụ thuộc random |
| `random_state` | 42 | Tái lập |
| `StandardScaler` | trước fit | 4 đặc trưng lệch đơn vị (giá ~4, DemandStd ~360) → phải chuẩn hóa, nếu không `DemandStd` áp đảo khoảng cách Euclid |

**Chỉ số nội bộ:**

- **Inertia/WCSS:** tổng bình phương khoảng cách tới tâm. Giảm đơn điệu theo k → **không dùng để chọn k trực tiếp**, chỉ tìm điểm gãy (Elbow).
- **Silhouette:** `s(i) = (b(i) − a(i)) / max(a(i), b(i))`. `a` = khoảng cách trung bình tới cụm mình, `b` = tới cụm gần nhất. Chi phí O(n²) nên chỉ chạy 3.271 sản phẩm.
- **Đánh đổi thực tế:** silhouette chọn k=2, dự án chọn k=3. Không có "đúng tuyệt đối" — đây là chọn theo mục đích phân khúc.

### 15.3. Apriori (`mining.py`)

**Mục tiêu:** tìm itemset phổ biến rồi sinh luật `A → B`.

**Cơ chế 2 pha:**

1. **Sinh itemset phổ biến**, dựa trên tính chất Apriori: nếu itemset có support < ngưỡng thì mọi superset cũng vậy → cắt tỉa.
2. **Sinh luật:** với mỗi itemset kích thước ≥2, thử mọi cách chia `A | rest`, tính confidence, giữ nếu ≥ ngưỡng.

**Chi tiết cài đặt:**

| Thành phần | Cách làm |
|---|---|
| Basket | `groupby('InvoiceNo') → set(StockCode)` |
| Candidate k=2 | Chỉ sinh cặp từ item phổ biến **trong từng basket**, không quét toàn bộ C(n,2) → giảm từ ~11 triệu xuống vài trăm nghìn phép thử |
| Candidate k≥3 | `combinations(present, k)`, kiểm tra mọi (k−1)-subset đã phổ biến |
| Lưu support | Dict `frozenset → support`, dùng lại khi tính confidence |

**Tham số:** `min_support=0,005`, `min_confidence=0,15`, `max_len=2`, ngưỡng diễn giải `support≥0,01`.

**Chỉ số nội bộ:** support (độ phổ biến), confidence (độ tin cậy có điều kiện), lift (so với độc lập). Lift tính bằng `confidence / support(consequent)` — không dùng công thức `support(A∪B)/(support(A)·support(B))` nhưng tương đương.

### 15.4. Isolation Forest (`mining.py`)

**Ý tưởng:** điểm bất thường dễ bị "cô lập" bởi cây chia ngẫu nhiên → đường đi ngắn.

**Cơ chế:**

1. Chọn ngẫu nhiên đặc trưng + ngưỡng chia, đệ quy đến khi điểm đơn độc.
2. Điểm bất thường: `s(x) = 2^(−E[h(x)] / c(n))`, với `h` = độ sâu trung bình, `c(n)` = độ sâu kỳ vọng của cây ngẫu nhiên.
3. `s → 1`: bất thường; `s → 0,5`: bình thường.

**Tham số dự án:**

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Đầu vào | `log1p(TotalQuantity, TotalRevenue, InvoiceCount)` | log để giảm ảnh hưởng đuôi phải |
| `contamination` | 0,02 | Ép 2% bị gắn cờ → 13/604 ngày |
| `random_state` | 42 | |
| Số cây | mặc định 100 | |

**Hạn chế:** `contamination` là giả định, không phải phát hiện. Nếu thực tế chỉ 0,5% bất thường, mô hình vẫn gắn 2% → 3/4 cờ là nhiễu.

### 15.5. Linear Regression

**Cơ chế:** OLS, cực tiểu `Σ(y − Xβ)²`. Nghiệm đóng `β = (XᵀX)⁻¹Xᵀy`.

**Trong dự án:** chạy sau `StandardScaler` + `SimpleImputer(median)`, one-hot `ProductCluster`.

**Vấn đề:** quan hệ nhu cầu–lag phi tuyến + zero-inflation → R² validation 0,0557, WAPE 1,873. Dùng làm **baseline tham chiếu**, không phải ứng viên.

### 15.6. Random Forest

**Cơ chế:** bagging (bootstrap) + chia node theo tiêu chí giảm MSE, mỗi node chỉ xét ngẫu nhiên `max_features` biến.

**Tham số:**

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `n_estimators` | 120 | Số cây |
| `min_samples_leaf` | 3 | Chống overfit từng sản phẩm |
| `max_features` | 0,8 | 80% biến mỗi lần chia → giảm tương quan giữa cây |
| `n_jobs` | −1 | Song song |

**Vấn đề:** cây hồi quy MSE rất nhạy outlier; nhu cầu max 19.152 kéo lá lệch. Kết quả tệ nhất nhóm (R² −0,242).

### 15.7. HistGradientBoosting (HGB)

**Cơ chế:** gradient boosting kiểu LightGBM —

1. Rời rạc hóa đặc trưng thành **histogram** (mặc định 255 bin) → fit nhanh trên 250k dòng.
2. Mỗi vòng: tính gradient, fit cây lá mới, cộng dồn với learning rate.
3. Cây phát triển **leaf-wise** (`max_leaf_nodes`), không theo depth.

**Tham số:**

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `loss` | `absolute_error` | **Quan trọng**: tối ưu MAE, không phải MSE → chống outlier. Với MAE, gradient = `sign(y−ŷ)` |
| `max_iter` | 250 | Số vòng boosting |
| `max_leaf_nodes` | 31 | Giới hạn phức tạp mỗi cây |
| `l2_regularization` | 2,0 | Phạt trọng số lá |
| `learning_rate` | mặc định 0,1 | |

**Vì sao `absolute_error`:** nhu cầu có đuôi cực dài; MSE sẽ dồn mô hình vào các ngày bán khủng, bỏ qua 84,8% ngày 0. MAE khớp với metric đánh giá (MAE).

### 15.8. XGBoost

**Cơ chế:** gradient boosting bậc 2, tối ưu hàm mục tiêu có regularization

```
Obj = Σ l(y, ŷ) + Σ [γT + ½λ||w||²]
```

dùng đạo hàm bậc 1 và bậc 2, chia node theo "gain".

**Tham số:** 250 cây, `max_depth=7`, `lr=0,05`, `subsample=0,8`, `colsample_bytree=0,8`, `objective=reg:squarederror` (tối ưu MSE).

**Vì sao kém ở đây:** objective MSE + depth 7 + 250 cây → học thuộc các đỉnh nhu cầu lớn, trong khi 84,8% quan sát bằng 0. MAE validation 5,14.

### 15.9. TwoStageDemand (hurdle) — mô hình được chọn

**Cơ chế 2 tầng, giải quyết zero-inflation:**

```
Tầng 1: phân loại  P(y > 0 | X)         → HGBClassifier, log-loss
Tầng 2: hồi quy    E[y | y > 0, X]      → HGBRegressor, MAE loss, fit trên y>0
Dự báo:  ŷ = 1{p ≥ τ} × max(0, reg.predict(X))
```

**Tham số:**

| Tham số | Giá trị |
|---|---|
| Classifier `max_iter` | 180 |
| Classifier `max_leaf_nodes` | 31 |
| Regressor `max_iter` | 250 |
| Regressor `loss` | `absolute_error` |
| `l2_regularization` | 2,0 cả hai |
| Ngưỡng τ | quét 0,10→0,90 bước 0,05, chọn MAE validation nhỏ nhất → **0,55** |

**Vì sao thắng:**

- Tầng 1 học "hôm nay có bán không" — tín hiệu mạnh hơn nhiều so với dự đoán lượng chính xác.
- Tầng 2 chỉ học trên ngày có bán → không bị 84,8% số 0 làm lệch.
- Ngưỡng τ tách biệt "xác suất" khỏi "lượng", cho phép điều chỉnh theo chi phí nghiệp vụ sau này.

**Nhược điểm:** mất thông tin kỳ vọng. `p × pos_hat` (không ngưỡng) có MAE 2,986 — kém hơn 2,908 một chút, nhưng về lý thuyết đúng hơn nếu tối ưu chi phí kỳ vọng.

### 15.10. Thành phần phụ trợ

| Thành phần | Cơ chế | Vai trò |
|---|---|---|
| `SimpleImputer(median)` | Điền median cột | `Lag_*` NaN ở những ngày đầu chuỗi; median an toàn hơn mean với đuôi dài |
| `SimpleImputer(most_frequent)` | Điền mode | `ProductCluster` |
| `StandardScaler` | `z = (x−μ)/σ` | Cần cho LinearRegression (hệ số ổn định); không cần cho cây nhưng vô hại |
| `OneHotEncoder(handle_unknown='ignore')` | One-hot | `ProductCluster` ∈ {0,1,2}; `ignore` tránh lỗi nếu test có cụm lạ |
| `ColumnTransformer` | Ghép pipeline numeric + categorical | |

---

## 16. Chỉ số và siêu tham số

### 16.1. Chỉ số đánh giá dự báo (`modeling.py::metrics`)

| Chỉ số | Công thức | Đọc thế nào | Số thực tế |
|---|---|---|---|
| **MAE** | `mean(|y−ŷ|)` | Sai số trung bình, cùng đơn vị với nhu cầu (sản phẩm/ngày) | Test 3,8807 → lệch 3,88 đơn vị/ngày/SP |
| **RMSE** | `√mean((y−ŷ)²)` | Phạt nặng sai số lớn; luôn ≥ MAE | Test 24,79 → có vài ngày lệch cực lớn |
| **WAPE** | `Σ|y−ŷ| / Σ|y|` | Sai số tương đối trên **tổng** nhu cầu. 0,1 = lệch 10%. **>1 = vô dụng** | Test 1,0782 → sai nhiều hơn tổng thực |
| **R²** | `1 − SS_res/SS_tot` | % biến thiên giải thích được. **Âm = tệ hơn đoán bằng trung bình** | Test 0,1343; baseline −0,6852 |

**Vì sao WAPE/R² ở đây "lạ":**

- RMSE (24,79) ≫ MAE (3,88) — gấp 6,4 lần. Nguyên nhân: phân phối lệch phải cực mạnh (max 19.152, median 0). Vài sản phẩm chủ lực dự báo sai vài nghìn đơn vị chi phối RMSE.
- WAPE > 1 là bất thường so với bài toán thông thường (thường < 0,3). Do 84,8% quan sát = 0, mẫu số `Σ|y|` nhỏ, mỗi lần dự báo dương sai đều bị tính vào tử số.
- R² 0,134 thấp nhưng **dương** → mô hình có tín hiệu; R² baseline âm → Lag-1 còn tệ hơn dùng trung bình. Đây là lý do TwoStageDemand vẫn "thắng" dù WAPE xấu.

MAE/RMSE/WAPE/R² **không** dùng để chọn mô hình — chọn theo MAE validation; test chỉ để báo cáo.

### 16.2. Chỉ số phân cụm

| Chỉ số | Ý nghĩa | Số thực tế |
|---|---|---|
| **Inertia (WCSS)** | Tổng bình phương khoảng cách tới tâm cụm. Càng nhỏ càng "chặt", nhưng luôn giảm khi tăng k → dùng để tìm **Elbow** | k=2: 9.386 → k=3: 6.969 → k=8: 2.361 |
| **Silhouette** | `(b−a)/max(a,b)`, a = khoảng cách nội cụm, b = cụm gần nhất. ∈ [−1,1]; >0,5 tách tốt | k=2: 0,625; **k=3: 0,5464** |

**Điểm cần giải thích khi bảo vệ:** silhouette cao nhất ở k=2 (0,625), nhưng chọn k=3. Lý do: k=2 chỉ tách "bán ít / bán nhiều", mất nhóm Chủ lực 47 sản phẩm (1,44%) — nhóm quan trọng nhất cho tồn kho. Elbow gãy mạnh ở k=3. Đây là chọn theo nghiệp vụ, không theo argmax silhouette.

### 16.3. Chỉ số luật kết hợp (Apriori)

| Chỉ số | Công thức | Đọc | Ví dụ thực tế |
|---|---|---|---|
| **Support** | `số hóa đơn chứa itemset / tổng hóa đơn` | Độ phổ biến. Thấp = hiếm, dễ nhiễu | `{22746, 22745}`: 1,17% (≈463/39.536 hóa đơn) |
| **Confidence** | `sup(A∪B)/sup(A)` | Xác suất có B khi đã có A | Có 22746 → 81,74% có 22745 |
| **Lift** | `conf(A→B)/sup(B)` | So với ngẫu nhiên. =1 độc lập; >1 dương | 45,97 → đồng mua gấp 46 lần ngẫu nhiên |

**Bẫy:** lift 45,97 rất ấn tượng nhưng support chỉ 1,17%. Sản phẩm hiếm + bán theo bộ → lift cao giả tạo. Vì vậy `association_interpretation.md` chỉ xếp hạng các luật có support ≥ 1% (932/5.187 luật).

### 16.4. Chỉ số bất thường

| Quy tắc | Ngưỡng | Ý nghĩa |
|---|---|---|
| **Rolling median ratio** | `x_t / median(x_{t−28..t−1}) ≥ 3` hoặc `≤ 0,33` | So hôm nay với trung vị 28 ngày trước. Dùng **median** + **shift 1** để không bị chính ngày đó làm lệch baseline |
| **IQR fence (Tukey)** | `Q3 + 1,5×IQR` | Chuẩn thống kê cổ điển. 1,5× là "outlier nhẹ"; 3× là "cực trị" |
| **Contamination** | 2% | Tham số IsolationForest: kỳ vọng 2% dữ liệu là bất thường → 13 ngày |
| **Tỷ lệ gắn cờ** | 41.168/530.098 = **7,77%** | Con số **đáng ngờ**: quá cao cho "bất thường". Nguyên nhân: nhu cầu sản phẩm vốn đã đột biến tự nhiên. Cần hiệu chỉnh ngưỡng theo nghiệp vụ |

Ba phương pháp không thay thế nhau: (a) bắt cú sốc toàn hệ thống, (b) bắt kết hợp bất thường nhiều chiều, (c) bắt đột biến riêng từng sản phẩm.

### 16.5. Chỉ số dữ liệu / mô hình

| Chỉ số | Giá trị | Nghĩa |
|---|---|---|
| **Zero-inflation** | 84,8% product-days = 0; ~78–80% mỗi tập | Tỷ lệ quan sát không bán. Là lý do tồn tại mô hình hurdle |
| **Concentration** | 10% SP = 63,17% doanh thu; 3 nước = 90,95% | Mức tập trung Pareto. Dự báo sai nhóm đầu gây thiệt hại lớn nhất |
| **Lag-1 baseline** | MAE 5,2342; WAPE 1,4543; R² −0,6852 | Mốc tối thiểu: "hôm qua bán bao nhiêu thì mai bán bấy nhiêu". Mô hình phải thắng mốc này mới có giá trị |
| **Số đặc trưng** | 23 | Trong đó 10 lag/rolling, 6 thời gian, 3 giá, 3 sản phẩm, 1 hóa đơn |
| **Tỷ lệ train dùng** | 250.000/1.193.915 = 20,9% | Sample ngẫu nhiên; ảnh hưởng khả năng học sản phẩm hiếm |
| **Số luật** | 5.187 sinh / 932 diễn giải | Ngưỡng sinh khác ngưỡng kể chuyện — phải tách |

### 16.6. Siêu tham số quyết định hành vi thuật toán

Đây là các con số **quyết định hành vi thuật toán**, phân biệt với metric đánh giá:

| File | Tham số | Giá trị | Nếu đổi thì sao |
|---|---|---|---|
| `preprocess` | `keep='first'` khi dedup | — | `last` giữ bản ghi sau |
| `feature_engineering` | `MIN_ACTIVE_DAYS` | 10 | Tăng → ít sản phẩm hơn, ít nhiễu hơn, mất sản phẩm hiếm |
| | `shift(1)` | bắt buộc | `shift(0)` = rò rỉ tương lai |
| | `min_periods` | 1 | Nhỏ → rolling có giá trị ngay từ đầu chuỗi nhưng kém tin cậy |
| | Cửa sổ rolling | 7/14/28 | 28 ≈ 1 tháng, khớp mùa vụ tuần |
| | IQR fence | 1,5× | 3× → ít cờ hơn |
| `clustering` | k | 3 | |
| | `n_init` | 10 | |
| `mining` | `min_support` | 0,005 | Tăng → ít luật, luật "chắc" hơn |
| | `min_confidence` | 0,15 | |
| | `max_len` | 2 | 3 → bùng nổ tổ hợp, chậm hơn nhiều |
| | `contamination` | 0,02 | Chính là số cờ IF |
| | Rolling median | 28, `min_periods=7` | |
| | Ratio ngưỡng | ≥3 / ≤0,33 | Đối xứng nghịch đảo |
| | Product anomaly | 90 quan sát, min 12, Q3+1,5IQR | |
| `modeling` | Sample train | 250.000 | |
| | τ | 0,55 | |
| | `max_iter` | 180/250 | |
| | `l2` | 2,0 | Tăng → mượt hơn, underfit |
| | `random_state` | 42 toàn bộ | Tái lập |

### 16.7. Metric ↔ thuật toán

| Thuật toán | Metric dùng để **fit/chọn** | Metric dùng để **báo cáo** |
|---|---|---|
| K-Means | inertia (fit), silhouette (chọn k) | silhouette |
| Apriori | support/confidence (ngưỡng lọc) | lift (xếp hạng) |
| IsolationForest | anomaly score, contamination | số điểm bị gắn cờ |
| Rolling median rule | ratio | số ngày bị gắn cờ |
| Linear/RF/HGB/XGB | MSE hoặc MAE loss | MAE, RMSE, WAPE, R² |
| TwoStage | log-loss (tầng 1), MAE (tầng 2), MAE validation (chọn τ) | MAE, RMSE, WAPE, R² |
| Baseline Lag-1 | không fit | MAE, RMSE, WAPE, R² |

### 16.8. Ba điểm dễ bị hỏi khi bảo vệ

1. **Vì sao HGB dùng `absolute_error` mà XGBoost dùng `squarederror`?** Vì metric chọn mô hình là MAE và dữ liệu có outlier cực đại. Đây là bất đối xứng có chủ đích, không phải sơ suất — nhưng nếu muốn so sánh công bằng thì phải cùng loss.
2. **Vì sao silhouette thấp (0,546) mà vẫn nhận?** Vì 4 đặc trưng có tương quan mạnh và phân phối lệch; silhouette > 0,5 được coi là cấu trúc cụm hợp lý. Cần nêu rõ đã chọn k theo nghiệp vụ.
3. **Vì sao không dùng Poisson/NegBin?** Chưa thử. Đó là hướng phát triển hợp lý nhất về mặt lý thuyết cho dữ liệu đếm zero-inflated, và nên nêu là hạn chế chứ không phải lựa chọn đã cân nhắc.

---

## 17. Phụ lục: định nghĩa chỉ số

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| MAE | `mean(|y − ŷ|)` | Sai số tuyệt đối trung bình, cùng đơn vị với nhu cầu |
| RMSE | `sqrt(mean((y − ŷ)²))` | Phạt nặng sai số lớn; nhạy với outlier |
| WAPE | `Σ|y − ŷ| / Σ|y|` | Sai số tương đối theo tổng; WAPE > 1 nghĩa là sai nhiều hơn tổng nhu cầu |
| R² | `1 − SS_res / SS_tot` | Tỷ lệ biến thiên được giải thích; âm nghĩa là tệ hơn dự báo bằng trung bình |
| Support | `|A ∪ B| / N` | Tỷ lệ hóa đơn chứa cả hai phía luật |
| Confidence | `support(A ∪ B) / support(A)` | Xác suất có B khi đã có A |
| Lift | `confidence(A→B) / support(B)` | Mức đồng xuất hiện so với ngẫu nhiên; > 1 là dương |
| Silhouette | `(b − a) / max(a, b)` | Độ tách cụm; gần 1 là tách tốt |

---

*Tài liệu phản ánh trạng thái mã nguồn và báo cáo trong `reports/` tại thời điểm 2026-10-20. Số liệu lấy trực tiếp từ các file JSON, không nhập tay.*
