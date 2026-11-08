# CẨM NANG ÔN TẬP & VẤN ĐÁP CHI TIẾT ĐỒ ÁN KHAI PHÁ DỮ LIỆU (KPDL)
**Đề tài:** Khai phá dữ liệu giao dịch bán lẻ và Dự báo nhu cầu tồn kho trên bộ dữ liệu *Online Retail II*

---

## PHẦN 1: SƠ ĐỒ LUỒNG CHẠY TOÀN BỘ DỰ ÁN (TỪ FILE GỐC ĐẾN KẾT QUẢ CUỐI)

Khi thầy cô hỏi: *"Em hãy trình bày luồng chạy của chương trình từ đầu đến cuối, file nào chạy trước, file nào chạy sau, đầu vào và đầu ra là gì?"*, bạn hãy nhìn vào sơ đồ 9 bước theo đúng thứ tự dưới đây:

```text
[DỮ LIỆU GỐC] online_retail_II.xlsx (1.067.371 dòng, 2 sheet Excel)
       │
       ▼
(Bước 1) preprocess.py ──► Làm sạch & tạo lưới sản phẩm-ngày
       ├──► data/processed/transactions_clean.parquet (.csv)     [1.007.913 dòng giao dịch sạch]
       ├──► data/processed/product_daily_demand.parquet (.csv)   [3.497.687 dòng Product × Day]
       └──► reports/preprocessing_report.json & preprocessing_removals.csv
       │
       ├──────────────────────────────────────────────────────────┐
       ▼                                                          ▼
(Bước 2) eda.py ──► Phân tích khám phá                     (Bước 3) mining.py ──► Luật kết hợp & Bất thường
  ├──► reports/eda_report.json                               ├──► reports/association_rules.json (.csv) [5.187 luật]
  ├──► reports/monthly_sales.csv, weekly_sales.csv           ├──► reports/association_interpretation.md [932 luật ≥1%]
  ├──► reports/product_summary.csv, customer_summary.csv     ├──► reports/anomaly_report.json & daily_anomalies.csv
  ├──► reports/country_summary.csv                           └──► reports/product_demand_anomalies.parquet [41.168 cờ]
  └──► reports/figures/ (5 biểu đồ doanh thu/sản lượng)
       │
       ▼
(Bước 4) feature_engineering.py ──► Lọc 3.271 SKU, tạo 23 biến đầu vào (Lag/Rolling), chia Train/Val/Test
       ├──► data/model/train.parquet         [1.193.915 dòng: 01/12/2009 → 30/11/2010]
       ├──► data/model/val.parquet           [294.390 dòng:   01/12/2010 → 28/02/2011]
       ├──► data/model/test.parquet          [928.964 dòng:   01/03/2011 → 09/12/2011]
       ├──► data/model/features_full.parquet [2.417.269 dòng]
       └──► reports/feature_engineering_report.json
       │
       ├─────────────────────────────┬────────────────────────────┬────────────────────────────┐
       ▼                             ▼                            ▼                            ▼
(Bước 5) clustering.py      (Bước 6) modeling.py         (Bước 7) intermittent_models.py  (Bước 8) weekly_forecasting.py
  K-Means (k=3) trên Train    Dự báo Ngày (Hurdle & ML)    Chuỗi gián đoạn Croston/SBA/TSB  Gom nhóm Dự báo Tuần (7 ngày)
  ├─► product_clusters.parquet├─► test_predictions.parquet ├─► intermittent_test_...parquet ├─► weekly_train/val/test.parquet
  └─► clustering_report.json  ├─► modeling_report.json     ├─► intermittent_models_...json  ├─► weekly_modeling_report.json
                              └─► demand_forecast_test.png └─► intermittent_models_...png   └─► weekly_model_comparison.png
                                     │                            │
                                     └──────────────┬─────────────┘
                                                    ▼
                                     (Bước 9) inventory_cost.py ──► Tối ưu Chi phí Tồn kho Newsvendor
                                       ├──► reports/inventory_cost_report.json
                                       ├──► reports/figures/inventory_cost_scenarios.png
                                       └──► reports/figures/service_level_tradeoff.png
```

---

## PHẦN 2: MỔ XẺ CHI TIẾT TỪNG FILE CODE — LUỒNG XỬ LÝ, THUẬT TOÁN, CHỈ SỐ & HÀNH ĐỘNG TIẾP THEO

---

### FILE 1: `preprocess.py` — Làm sạch dữ liệu & Tạo lưới thời gian

#### 1. File tương ứng
- **Đầu vào:** `online_retail_II.xlsx` (`1.067.371` dòng từ 2 sheet `Year 2009-2010` và `Year 2010-2011`).
- **Hàm xử lý chính:** `find_column()`, `load_workbook()`, `main()`.
- **Đầu ra dữ liệu:**
  - `data/processed/transactions_clean.parquet` (và `.csv`): Bảng giao dịch đã làm sạch (`1.007.913` dòng).
  - `data/processed/product_daily_demand.parquet` (và `.csv`): Bảng nhu cầu từng sản phẩm theo từng ngày (`3.497.687` dòng).
- **Đầu ra báo cáo:** `reports/preprocessing_report.json`, `reports/preprocessing_removals.csv`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Hợp nhất & Chuẩn hóa cột (`load_workbook`, `find_column`):** Đọc cả 2 sheet Excel, đổi tên cột về chuẩn thống nhất (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`), xóa khoảng trắng thừa, viết hoa `StockCode` và `Description`.
2. **Bộ lọc chất lượng 6 bước tuần tự (hàm `remove`):**
   - **Bước 1 (`cancelled_invoice`):** Kiểm tra `InvoiceNo.str.startswith("C")` $\to$ Loại **19.494 dòng** (`1.8264%`). Đây là các hóa đơn hủy đơn hàng.
   - **Bước 2 (`negative_quantity`):** Kiểm tra `Quantity < 0` $\to$ Loại **3.457 dòng** (`0.3239%`). Đây là các bút toán điều chỉnh kho/hàng lỗi.
   - **Bước 3 (`nonpositive_price`):** Kiểm tra `UnitPrice <= 0` $\to$ Loại **2.750 dòng** (`0.2576%`).
   - **Bước 4 (`exact_duplicate`):** Kiểm tra `duplicated(keep="first")` $\to$ Loại **33.757 dòng** (`3.1626%`) bị ghi trùng trong hệ thống.
   - **Bước 5 (`Description` thiếu):** Tìm tên mô tả xuất hiện nhiều nhất (`mode()`) của cùng mã `StockCode` để điền cho **4.382 dòng** bị thiếu `Description` (`0` dòng bị loại).
   - **Bước 6 (`CustomerID` thiếu):** Có **228.488 dòng** sạch không có `CustomerID` — thuật toán **giữ nguyên là `NA`** (không xóa vì khách vãng lai vẫn mua hàng thật).
3. **Tách mã dịch vụ & Điền ngày bằng 0 (Zero-filling / Cartesian Grid):**
   - Lọc bỏ **4.260 dòng** có `StockCode` không chứa chữ số (như phí ship `POST`, `BANK CHARGES`, `M`), còn lại **1.003.653 giao dịch hàng hóa** của **4.733 mã sản phẩm**.
   - Nhóm theo `(StockCode, Date)` để tính:
     $$\text{DailyDemand} = \sum \text{Quantity}, \quad \text{UnitPrice} = \text{mean}(\text{UnitPrice}), \quad \text{InvoiceCount} = \text{nunique}(\text{InvoiceNo})$$
   - Tạo lưới Đề-các `MultiIndex.from_product([4.733 sản phẩm, 739 ngày])` = **3.497.687 dòng**. Những ngày không có hóa đơn được gán `DailyDemand = 0`, `InvoiceCount = 0`, còn `UnitPrice = NaN` (không bịa giá cho ngày không bán).

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Chỉ số trong `preprocessing_report.json` | Giá trị | Ý nghĩa & Hành động tiếp theo (Next Action) |
| :--- | :---: | :--- |
| `total_removed_by_quality_steps` | **59.458 dòng (5.57%)** | Tỷ lệ nhiễu vừa phải (<10%), chứng tỏ bộ dữ liệu gốc đủ tin cậy để đưa vào khai phá. |
| `missing_customer_id_after_cleaning` | **228.488 dòng (22.67%)** | Cứ 4 giao dịch thì có gần 1 giao dịch không rõ mã khách hàng $\Rightarrow$ **Quyết định:** Chỉ dùng bảng khách hàng cho EDA (`eda.py`), **tuyệt đối không dùng `CustomerID` làm biến đầu vào khi dự báo nhu cầu sản phẩm**. |
| Tỷ lệ ngày `DailyDemand = 0` | **2.967.589 / 3.497.687 (84.84%)** | Phát hiện dữ liệu bị **Lạm phát số 0 (Zero-Inflated / Intermittent)** cực nặng $\Rightarrow$ **Quyết định:** Ở bước Feature Engineering phải lọc bớt sản phẩm quá thưa và ở bước Modeling phải dùng mô hình 2 pha (Hurdle) thay vì hồi quy thường. |

---

### FILE 2: `eda.py` — Phân tích Khám phá Dữ liệu

#### 1. File tương ứng
- **Đầu vào:** `data/processed/transactions_clean.parquet` (`1.007.913` dòng).
- **Đầu ra báo cáo & dữ liệu:**
  - `reports/eda_report.json`: Tổng hợp toàn bộ chỉ số phân phối.
  - `reports/monthly_sales.csv`, `reports/weekly_sales.csv`, `reports/product_summary.csv`, `reports/customer_summary.csv`, `reports/country_summary.csv`.
- **Đầu ra biểu đồ (`reports/figures/`):**
  - `monthly_quantity.png`, `monthly_revenue.png`, `weekly_quantity.png`, `country_revenue_top10.png`, `customer_revenue_distribution.png`.

#### 2. Thuật toán xử lý bên trong như thế nào?
- Gom nhóm (`groupby`) theo 4 chiều:
  1. **Theo Ngày / Tuần (`resample('W-MON')`) / Tháng (`resample('MS')`):** Tính tổng `Quantity`, `Revenue`, số lượng hóa đơn để tìm quy luật mùa vụ.
  2. **Theo Sản phẩm (`StockCode`):** Xếp hạng doanh thu từ cao xuống thấp, tính tỷ trọng doanh thu của Top 10% sản phẩm đầu bảng.
  3. **Theo Khách hàng (`CustomerID`):** Tính tần suất mua (`InvoiceCount`), tổng chi tiêu (`Revenue`) và các mốc phân vị (`25%, 50%, 75%, 90%, 99%`).
  4. **Theo Quốc gia (`Country`):** Tính thị phần doanh thu (`RevenueShare`) của từng nước trong 43 quốc gia.

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Chỉ số trong `eda_report.json` | Giá trị | Ý nghĩa & Hành động tiếp theo (Next Action) |
| :--- | :---: | :--- |
| `top_10pct_products_revenue_share` | **63.17%** | 10% mặt hàng mang lại gần 2/3 doanh thu $\Rightarrow$ **Quyết định:** Cần phân cụm sản phẩm (`clustering.py`) để tách riêng nhóm hàng Chủ lực/Bán chạy khỏi nhóm hàng bán chậm. |
| `country_revenue_concentration_top_3` | **~90% (UK chiếm >84%)** | Doanh thu tập trung tuyệt đối ở Anh (UK) $\Rightarrow$ **Quyết định:** Khi tạo biến ngày nghỉ lễ (`IsHoliday`) ở `feature_engineering.py`, sử dụng lịch nghỉ lễ của Vương quốc Anh (`holidays.UK`). |
| `monthly_quantity_peak` / `monthly_revenue_peak` | **Tháng 11 (Nov)** | Đỉnh điểm mua sắm rơi vào tháng 9–11 (mùa lễ hội cuối năm) $\Rightarrow$ **Quyết định:** Phải đưa các biến chu kỳ thời gian (`Month`, `WeekOfYear`, `DayOfWeek`) vào mô hình dự báo. |

---

### FILE 3: `feature_engineering.py` — Tạo đặc trưng & Chia tập Train/Val/Test

#### 1. File tương ứng
- **Đầu vào:** `data/processed/product_daily_demand.parquet` (`3.497.687` dòng).
- **Hàm xử lý chính:** `add_holiday_flag()`, `filter_low_frequency()`, `flag_outliers()`, `build_history_features()`, `build_price_features()`, `transform_target()`, `build_product_features()`.
- **Đầu ra dữ liệu (`data/model/`):**
  - `train.parquet` (`1.193.915` dòng: `01/12/2009` $\to$ `30/11/2010`)
  - `val.parquet` (`294.390` dòng: `01/12/2010` $\to$ `28/02/2011`)
  - `test.parquet` (`928.964` dòng: `01/03/2011` $\to$ `09/12/2011`)
  - `features_full.parquet` (`2.417.269` dòng)
- **Đầu ra báo cáo:** `reports/feature_engineering_report.json`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Lọc sản phẩm quá thưa (`filter_low_frequency`):**
   - Đếm số ngày có `DailyDemand > 0` của từng sản phẩm **chỉ trong giai đoạn Train (`Date <= 2010-11-30`)**.
   - Nếu sản phẩm bán `< 10 ngày` trong cả năm đầu tiên $\to$ Loại bỏ (**loại 1.462 sản phẩm**, giữ lại **3.271 sản phẩm** tương ứng `2.417.269` dòng).
2. **Tạo đặc trưng lịch sử chống rò rỉ dữ liệu (`build_history_features`):**
   - Với mỗi sản phẩm, dịch lùi chuỗi nhu cầu đi $k$ ngày (`shift(k)`):
     - $\text{Lag\_1}_t = y_{t-1}, \quad \text{Lag\_7}_t = y_{t-7}, \quad \text{Lag\_14}_t = y_{t-14}, \quad \text{Lag\_28}_t = y_{t-28}$
   - Tạo chuỗi đã dịch 1 ngày $s_t = y_{t-1}$, sau đó mới tính trượt (`rolling`) trên $s_t$:
     - $\text{RollingMean\_w}_t = \frac{1}{w}\sum_{j=1}^{w} y_{t-j}$ (với $w \in \{7, 14, 28\}$)
     - $\text{RollingSum\_7}_t = \sum_{j=1}^{7} y_{t-j}$, $\text{RollingStd\_28}_t = \text{std}(y_{t-1}, \dots, y_{t-28})$
   - Tính `DaysSinceLastSale`: Khoảng cách ngày từ ngày $t$ đến ngày gần nhất trước đó có $y > 0$.
3. **Tạo đặc trưng giá trễ (`build_price_features`):**
   - Điền xuôi (`ffill`) giá của lần bán gần nhất qua các ngày không bán, rồi dịch lùi 1 ngày để lấy `PriceLag_1` (giá ngày hôm qua) và `PriceChange = PriceLag_1 - PriceLag_2`, kèm `InvoiceCountLag_1` (số hóa đơn ngày hôm qua).
4. **Chia tập theo thời gian & Gắn đặc trưng sản phẩm từ Train (`build_product_features`):**
   - Chia bảng thành `train`, `val`, `test` theo mốc ngày.
   - Chỉ dùng bảng `train` để tính `TotalActiveDays`, `TotalQuantitySold`, `AvgUnitPrice` và chạy K-Means ($k=3$) tạo cột `ProductCluster`, sau đó ánh xạ (`map`) sang `val` và `test`.

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Chỉ số trong `feature_engineering_report.json` | Giá trị | Ý nghĩa & Hành động tiếp theo (Next Action) |
| :--- | :---: | :--- |
| `products_after` / `products_removed` | **3.271 giữ lại / 1.462 loại** | Giúp giảm 1,08 triệu dòng toàn số 0 của các mặt hàng rác/ngừng bán, làm sạch không gian học cho mô hình. |
| Phân vị `25%, 50%, 75%` của `DailyDemand` ở cả Train/Val/Test | **Đều bằng `0.0`** | Tới phân vị thứ 75% mà nhu cầu vẫn bằng 0 (vì ~79–82% ngày bằng 0), nhưng `max` lên tới `19.152` $\Rightarrow$ **Quyết định:** Không thể dùng hồi quy thông thường hay phân vị 0.75 vô điều kiện; bắt buộc dùng mô hình 2 pha ở `modeling.py`. |

---

### FILE 4: `clustering.py` — Phân cụm sản phẩm bằng K-Means

#### 1. File tương ứng
- **Đầu vào:** `data/model/train.parquet` (`1.193.915` dòng của `3.271` sản phẩm).
- **Hàm xử lý chính:** `aggregate_product_features()`, `evaluate_k_range()`, `fit_final_clusters()`.
- **Đầu ra dữ liệu:** `data/processed/product_clusters.parquet` (và `.csv`).
- **Đầu ra báo cáo:** `reports/clustering_report.json`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Tổng hợp 4 đặc trưng trên tập Train cho mỗi mã hàng $i \in \{1 \dots 3271\}$:**
   - $x_{i,1} = \text{AvgDailyDemand}$ (Trung bình lượng bán/ngày)
   - $x_{i,2} = \text{ActiveDays}$ (Tổng số ngày có bán $>0$)
   - $x_{i,3} = \text{AvgUnitPrice}$ (Giá bán trung bình)
   - $x_{i,4} = \text{DemandStd}$ (Độ lệch chuẩn lượng bán)
2. **Chuẩn hóa Z-score (`StandardScaler`):**
   $$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$
   *(Bắt buộc phải chuẩn hóa vì `DemandStd` có giá trị hàng trăm trong khi `AvgUnitPrice` chỉ vài bảng Anh; nếu không chuẩn hóa, khoảng cách Euclid trong K-Means sẽ bị biến có đơn vị lớn chi phối hoàn toàn).*
3. **Thử nghiệm $k \in \{2, 3, \dots, 8\}$ và tính 2 chỉ số:**
   - **Inertia (WCSS - Tổng bình phương khoảng cách trong cụm):** $\sum_{c=1}^k \sum_{x \in C_c} \|x - \mu_c\|^2$.
   - **Silhouette Score:** $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$, trong đó $a(i)$ là khoảng cách trung bình tới các điểm cùng cụm, $b(i)$ là khoảng cách trung bình tới cụm gần nhất tiếp theo.
4. **Sắp xếp lại nhãn cụm tăng dần theo nhu cầu (`remap`):** Đảm bảo Cụm `0` luôn là nhóm bán ít nhất và Cụm `2` luôn là nhóm bán nhiều nhất.

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Chỉ số trong `clustering_report.json` | Giá trị | Ý nghĩa & Hành động tiếp theo (Next Action) |
| :--- | :---: | :--- |
| `silhouette_scores` ($k=2 \to 8$) | $k=2$: `0.6250`<br>**$k=3$: `0.5464`**<br>$k=4$: `0.5557` | Chọn **$k=3$** vì $\text{Silhouette} = 0.5464 > 0.50$ (cấu trúc phân cụm mạnh), Inertia giảm mạnh từ `9.386` xuống `6.969`, và chia đúng 3 nhóm chuẩn quản trị kho ABC. |
| Hồ sơ 3 cụm (`cluster_profiles`) | **Cụm 0 (Phổ thông):** 2.548 SKU (77.9%)<br>**Cụm 1 (Bán chạy):** 676 SKU (20.7%)<br>**Cụm 2 (Chủ lực):** 47 SKU (1.4%) | **Hành động tiếp theo:**<br>1. Đưa nhãn `ProductCluster` (One-Hot Encoding) vào làm biến dự báo trong `modeling.py` và `weekly_forecasting.py`.<br>2. Áp dụng chính sách tồn kho khác nhau: Cụm 1 & 2 cần dự báo sát và giữ tồn kho an toàn cao; Cụm 0 chỉ nhập hàng theo đơn hoặc dự báo gộp theo tuần. |

---

### FILE 5: `mining.py` — Khai phá Luật kết hợp Apriori & Phát hiện Bất thường

#### 1. File tương ứng
- **Đầu vào:** `data/processed/transactions_clean.parquet`.
- **Hàm xử lý chính:** `apriori_rules()`, `detect_anomalies()`.
- **Đầu ra báo cáo & dữ liệu:**
  - Luật kết hợp: `reports/association_rules.json`, `reports/association_rules.csv`, `reports/association_interpretation.md`.
  - Bất thường: `reports/anomaly_report.json`, `reports/daily_anomalies.csv`, `reports/product_demand_anomalies.parquet`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Thuật toán Apriori (`apriori_rules`):**
   - Gom các mặt hàng trong cùng một hóa đơn `InvoiceNo` thành một tập hợp (`set` / giỏ hàng), tổng cộng $N = 39.536$ giỏ hàng.
   - **Bước $k=1$:** Đếm tần suất từng mã hàng, giữ lại các mã xuất hiện trong $\ge 0.5\%$ số hóa đơn (`min_support = 0.005`).
   - **Bước $k=2$:** Chỉ ghép cặp các mã phổ biến ở bước 1 nằm trong cùng giỏ hàng, giữ lại các cặp có $\text{Support}(A \cup B) \ge 0.005$.
   - **Sinh luật $A \Rightarrow B$:** Với mỗi tập phổ biến, tính:
     $$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}, \quad \text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$
     Giữ các luật có $\text{Confidence} \ge 15\%$ và sắp xếp giảm dần theo `Lift`.
2. **Thuật toán Phát hiện bất thường (`detect_anomalies`):**
   - **Nhánh 1 (Thống kê trượt cấp cửa hàng):** Tính trung vị trượt 28 ngày trước đó (`rolling(28).median().shift(1)`). Nếu ngày $t$ có tổng sản lượng hoặc doanh thu gấp $\ge 3$ lần hoặc $\le 0.33$ lần trung vị $\to$ Gắn cờ bất thường.
   - **Nhánh 2 (Isolation Forest cấp cửa hàng):** Biến đổi $\log(1+x)$ cho 3 biến `(TotalQuantity, TotalRevenue, InvoiceCount)` và chạy `IsolationForest(contamination=0.02, random_state=42)` (cô lập các điểm dị biệt bằng cây phân hoạch ngẫu nhiên).
   - **Nhánh 3 (Bất thường cấp SKU-ngày bằng Tukey IQR):** Trên chuỗi các ngày có bán ($Y>0$) của từng sản phẩm, tính $Q_1, Q_3$ của 90 lần bán trước đó (`shift(1).rolling(90, min_periods=12)`). Nếu $y_t > Q_3 + 1.5 \times (Q_3 - Q_1)$ $\to$ Gắn cờ `DemandAnomaly = True`.

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Chỉ số trong `association_rules.json` & `anomaly_report.json` | Giá trị | Ý nghĩa & Hành động tiếp theo (Next Action) |
| :--- | :---: | :--- |
| `rule_count` / `interpretable_rule_count_support_at_least_1pct` | **5.187 luật** / **932 luật có Support $\ge 1\%$** | Lọc lấy 932 luật có Support $\ge 1\%$ (xuất hiện ở $\ge 395$ hóa đơn) để tránh các luật ăn may của hàng hiếm. |
| Top luật: `22746` $\Rightarrow$ `22748` | Support: **1.20%**<br>Confidence: **84.04%**<br>Lift: **44.78** | **Hành động kinh doanh:** Thiết lập combo bán kèm (Bundle) hoặc gợi ý "Mua cùng nhau" trên website cho bộ *Poppy's Playhouse* (`22745, 22746, 22748`) và bộ gỗ Giáng sinh (`22578, 22579`). |
| `quantity_rule_anomalies` / `isolation_forest_anomalies` | **30 ngày** / **13 ngày** (trên 604 ngày) | Các ngày đỉnh điểm (như `27/09/2010` bán 125.489 sp) trùng với đơn đặt sỉ lớn. |
| `product_demand_anomalies` | **41.168 / 530.098 ngày bán (7.766%)** | **Quyết định quan trọng cho bước Modeling:** Vì 7.77% ngày bán có lượng mua vọt cao đột biến, khi huấn luyện mô hình dự báo **không được xóa các dòng này** (vì là cầu thật), nhưng **phải dùng hàm mất mát bậc nhất `L1 / absolute_error`** thay vì `L2 / squared_error` để mô hình không bị các đỉnh này kéo lệch. |

---

### FILE 6: `modeling.py` — Dự báo Nhu cầu Ngày & Mô hình 2 Giai đoạn Hurdle (`TwoStageDemand`)

#### 1. File tương ứng
- **Đầu vào:** `data/model/train.parquet`, `data/model/val.parquet`, `data/model/test.parquet`.
- **Đầu ra dữ liệu:** `data/model/predictions/test_predictions.parquet`, `test_predictions_sample.csv`.
- **Đầu ra báo cáo & biểu đồ:** `reports/modeling_report.json`, `reports/figures/demand_forecast_test.png`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Tiền xử lý trong Pipeline (`ColumnTransformer`):**
   - 22 biến số (`numeric`): Điền khuyết bằng trung vị (`SimpleImputer(strategy='median')`) + Chuẩn hóa `StandardScaler()`.
   - 1 biến phân loại (`ProductCluster`): Điền khuyết bằng `most_frequent` + Mã hóa `OneHotEncoder()`.
2. **Huấn luyện 4 mô hình 1 giai đoạn (Single-Stage):**
   - `LinearRegression`, `RandomForestRegressor`, `HistGradientBoostingRegressor(loss='absolute_error')`, `XGBRegressor(objective='reg:squarederror')`.
3. **Huấn luyện Mô hình 2 giai đoạn `TwoStageDemand` (Hurdle Model):**
   - **Giai đoạn 1 (Phân loại Có bán / Không bán):** Gán nhãn nhị phân $z_i = \mathbb{I}(y_i > 0) \in \{0, 1\}$. Huấn luyện `HistGradientBoostingClassifier` để dự đoán xác suất bán $p_i = \hat{P}(y_i > 0 \mid X_i)$.
   - **Giai đoạn 2 (Hồi quy lượng bán khi có bán):** Lọc riêng các dòng có $y_i > 0$ (`positive_train`), huấn luyện `HistGradientBoostingRegressor(loss='absolute_error')` để dự đoán lượng bán dương $\hat{y}_i^+$.
   - **Tinh chỉnh ngưỡng cắt $\theta$ trên tập Validation:** Quét $\theta \in \{0.10, 0.15, \dots, 0.90\}$ cho công thức:
     $$\hat{y}_i = \begin{cases} \hat{y}_i^+ & \text{nếu } p_i \ge \theta \\ 0 & \text{nếu } p_i < \theta \end{cases}$$
     Chọn $\theta$ làm cực tiểu hóa MAE trên tập Validation $\to$ Tìm được **$\theta^* = 0.55$**.
4. **Lựa chọn mô hình & Đánh giá 1 lần trên tập Test:**
   - So sánh MAE trên tập Validation của cả 5 mô hình $\to$ Chọn `TwoStageDemand` (MAE Val thấp nhất = `2.9081`).
   - Áp dụng `TwoStageDemand` ($\theta^* = 0.55$) lên tập Test (`928.964` dòng) và so sánh với Baseline `Lag_1` ($\hat{y}_t = y_{t-1}$).

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
**Công thức 4 chỉ số đánh giá trong `modeling_report.json`:**
- **MAE (Sai số tuyệt đối trung bình):** $\frac{1}{N}\sum |y_i - \hat{y}_i|$ (càng nhỏ càng tốt).
- **RMSE (Căn bậc hai sai số toàn phương trung bình):** $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ (phạt nặng các sai số lớn).
- **WAPE (Tỷ lệ sai số tuyệt đối có trọng số):** $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$ (dưới $1.0$ là tổng lỗi nhỏ hơn tổng cầu; càng nhỏ càng tốt).
- **$R^2$ (Hệ số xác định):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ ($R^2 > 0$ là tốt hơn dự báo bằng trung bình; $R^2 < 0$ là tệ hơn trung bình).

| Mô hình (trong `modeling_report.json`) | Tập đánh giá | MAE | RMSE | WAPE | $R^2$ | Ý nghĩa & Quyết định sau khi đọc chỉ số |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `RandomForest` (L2) | Validation | 5.9235 | 29.93 | 1.9332 | -0.2424 | **Loại bỏ:** $R^2 < 0$, WAPE gần 200% do bị 80% số 0 và ngoại lai làm nhiễu. |
| `XGBoost` (L2) | Validation | 5.1426 | 32.76 | 1.6783 | -0.4885 | **Loại bỏ:** Tối ưu bình phương sai số (`reg:squarederror`) thất bại trên dữ liệu gián đoạn. |
| `LinearRegression` | Validation | 5.7385 | 26.10 | 1.8728 | 0.0557 | **Loại bỏ:** Không tách được rào cản số 0, ngày nào cũng dự báo dương nhỏ. |
| `HistGradientBoosting` (L1 - 1 pha) | Validation | 3.0573 | 26.99 | 0.9978 | -0.0099 | Nhờ hàm mất mát L1 nên WAPE giảm mạnh về ~1.0, nhưng $R^2$ vẫn âm nhẹ. |
| **`TwoStageDemand` ($\theta=0.55$)** | **Validation** | **2.9081** | **26.10** | **0.9491** | **0.0555** | **ĐƯỢC CHỌN LÀM CHAMPION CẤP NGÀY** vì đạt MAE và WAPE tốt nhất trên Validation. |
| `Lag1_Baseline` | **Test** | 5.2342 | 34.59 | 1.4543 | -0.6852 | Điểm chuẩn ngây thơ trên tập Test. |
| **`TwoStageDemand` ($\theta=0.55$)** | **Test** | **3.8807** | **24.79** | **1.0782** | **0.1343** | **Kết luận & Hành động tiếp theo:** Tốt hơn vượt trội so với Baseline (giảm 37.6% WAPE, tăng $R^2$ từ `-0.685` lên `+0.134`). **Tuy nhiên**, vì $\text{WAPE} = 1.0782 > 1.0$, chưa thể dùng trực tiếp dự báo ngày để tự động đặt hàng $\Rightarrow$ **Phải triển khai tiếp 3 bước nâng cấp (`intermittent_models.py`, `weekly_forecasting.py`, `inventory_cost.py`)!** |

---

### FILE 7: `intermittent_models.py` — Các Mô hình Chuỗi thời gian Gián đoạn (Croston, SBA, TSB)

#### 1. File tương ứng
- **Đầu vào:** `data/processed/product_daily_demand.parquet`, `data/model/test.parquet`.
- **Hàm xử lý chính:** `croston_forecast(series, alpha, sba)`, `tsb_forecast(series, alpha, beta)`.
- **Đầu ra dữ liệu:** `data/model/predictions/intermittent_test_predictions.parquet`, `reports/intermittent_predictions_sample.csv`.
- **Đầu ra báo cáo & biểu đồ:** `reports/intermittent_models_report.json`, `reports/figures/intermittent_models_comparison.png`.

#### 2. Thuật toán xử lý bên trong như thế nào?
Với mỗi chuỗi sản phẩm riêng biệt, tại ngày $t$:
1. **Thuật toán Croston (`croston_forecast` với `sba=False`):**
   - Gọi $z_t$ là ước lượng quy mô ngày có bán, $p_t$ là ước lượng khoảng cách số ngày giữa 2 lần bán.
   - Dự báo cho ngày $t$: $\hat{y}_t = \frac{z_{t-1}}{p_{t-1}}$.
   - Cập nhật sau khi biết $y_t$:
     - Nếu $y_t > 0$: $z_t = \alpha y_t + (1-\alpha)z_{t-1}$ và $p_t = \alpha(t - t_{\text{prev}}) + (1-\alpha)p_{t-1}$.
     - Nếu $y_t = 0$: Giữ nguyên $z_t = z_{t-1}, p_t = p_{t-1}$.
2. **Thuật toán SBA (`croston_forecast` với `sba=True`):**
   - Nhân thêm hệ số hiệu chỉnh chệch của Syntetos-Boylan: $\hat{y}_t = \left(1 - \frac{\alpha}{2}\right)\frac{z_{t-1}}{p_{t-1}}$.
3. **Thuật toán TSB (`tsb_forecast`):**
   - Thay $p_t$ bằng **xác suất có bán hàng trong ngày**, dự báo $\hat{y}_t = p_{t-1} \times z_{t-1}$.
   - Cập nhật ở **mọi ngày**:
     - Nếu $y_t > 0$: $p_t = \beta(1) + (1-\beta)p_{t-1}$ và $z_t = \alpha y_t + (1-\alpha)z_{t-1}$.
     - Nếu $y_t = 0$: $p_t = (1-\beta)p_{t-1}$ (xác suất giảm dần theo hàm mũ mỗi ngày không bán được hàng!).

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Mô hình (trong `intermittent_models_report.json`) | Tham số tối ưu (Val) | Test MAE | Test RMSE | Test WAPE | Test $R^2$ | Ý nghĩa & Quyết định sau khi đọc chỉ số |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Croston` | $\alpha = 0.05$ | 5.5567 | 26.23 | 1.5439 | 0.0311 | Giảm RMSE mạnh so với `Lag_1` (`26.23` vs `34.59`), nhưng WAPE cao (`1.5439`) vì không hạ dự báo về 0 khi hàng ngừng bán. |
| `SBA` | $\alpha = 0.20$ | 5.8030 | 34.21 | 1.6123 | -0.6484 | Nhạy với các cú nhảy số lượng ở tập Test. |
| **`TSB`** | $\alpha = 0.20, \beta = 0.40$ | **4.5276** | **25.72** | **1.2580** | **0.0681** | **Tốt nhất trong họ mô hình chuỗi thời gian đơn biến**, thắng cả `Lag_1` và `Croston`. |
| **`TwoStageDemand` (Đối chứng)** | $\theta = 0.55$ | **3.8807** | **24.79** | **1.0782** | **0.1343** | **Kết luận:** Chứng minh mô hình Học máy 2 pha (`TwoStageDemand`) vượt qua cả thuật toán thống kê TSB kinh điển (nhờ dùng thêm biến lịch, giá và cụm K-Means). Tuy nhiên ở cấp độ ngày, cả TSB lẫn TwoStageDemand đều có $\text{WAPE} > 1.0$ $\Rightarrow$ **Chuyển sang giải pháp gom nhóm theo Tuần ở `weekly_forecasting.py`**. |

---

### FILE 8: `weekly_forecasting.py` — Tổng hợp & Dự báo Nhu cầu theo Tuần

#### 1. File tương ứng
- **Đầu vào:** `data/processed/product_daily_demand.parquet`, `data/processed/product_clusters.parquet`, `data/model/predictions/test_predictions.parquet`.
- **Đầu ra dữ liệu:** `data/model/weekly_train.parquet` (`166.821` dòng), `weekly_val.parquet` (`39.252` dòng), `weekly_test.parquet` (`127.569` dòng).
- **Đầu ra báo cáo & biểu đồ:** `reports/weekly_modeling_report.json`, `reports/figures/weekly_model_comparison.png`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Đồng bộ quần thể & Gom nhóm tuần 7 ngày không tràn ranh giới:**
   - Giữ đúng **3.271 sản phẩm** có $\ge 10$ ngày bán ở tập Train.
   - Gom 7 ngày (Thứ Hai `WeekStart` $\to$ Chủ Nhật `WeekEnd`) thành 1 dòng:
     $$\text{WeeklyDemand} = \sum_{d \in \text{Week}} \text{DailyDemand}_d$$
   - Chỉ giữ các tuần đủ 7 ngày nằm trọn vẹn trong từng phân vùng (`WeekEnd <= 2010-11-30` cho Train; `2010-12-01 <= WeekStart` và `WeekEnd <= 2011-02-28` cho Val; `WeekStart >= 2011-03-01` và `WeekEnd <= 2011-12-04` cho Test) để **không có bất kỳ ngày nào bị rò rỉ qua ranh giới**.
2. **Tạo 15 đặc trưng cấp Tuần (`shift(>=1)` theo tuần):**
   - `Month`, `WeekOfYear`, `Lag_1W`, `Lag_2W`, `Lag_4W`, `Lag_8W`, `RollingMean_4W`, `RollingMean_8W`, `RollingStd_4W`, `RollingSum_4W`, `PriceLag_1W`, `InvoiceCountLag_1W`, `TrainActiveWeeks`, `TrainTotalSold`, `ProductCluster`.
3. **Huấn luyện & Đối chứng 8 phương án trên tập Tuần:**
   - `Lag1W_Baseline`, `Ridge`, `RandomForest`, `XGBoost`, `HistGB_L2`, **`HistGB_L1`** (`loss='absolute_error'`), **`TwoStageWeekly`** (Hurdle cấp tuần), và **`Daily_TwoStage_SummedToWeekly`** (lấy dự báo ngày của `TwoStageDemand` cộng gộp lên 39 tuần Test).

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Mô hình (trong `weekly_modeling_report.json`) | Val WAPE | Test MAE | Test RMSE | Test WAPE | Test $R^2$ | Ý nghĩa & Quyết định vận hành |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Lag1W_Baseline` | 1.0686 | 21.3067 | 87.78 | 0.8563 | 0.1316 | Chỉ cần gom lên tuần, ngay cả Baseline cũng giảm WAPE từ `1.454` (ngày) xuống `0.856` (tuần). |
| `Ridge` / `XGBoost` / `RandomForest` (L2) | 1.19–1.38 | 22.3–24.7 | 71.5–79.0 | 0.896–0.994 | 0.297–0.425 | Tối ưu L2 giúp $R^2$ tăng vọt lên `0.30–0.42`, nhưng WAPE vẫn quanh `0.90`. |
| `Daily_TwoStage_SummedToWeekly` | — | 17.8472 | **68.68** | 0.7173 | **0.4684** | Cộng gộp dự báo ngày lên tuần giúp sai số lệch ngày trong tuần tự triệt tiêu, WAPE giảm từ `1.078` xuống `0.7173`! |
| **`TwoStageWeekly` (Hurdle Tuần)** | 0.8259 | 17.0064 | 69.70 | **0.6835** | **0.4524** | Mô hình 2 pha huấn luyện trực tiếp ở cấp tuần. |
| **`HistGB_L1` (Champion Tuần)** | **0.7932** | **16.3357** | 71.10 | **0.6565** | **0.4302** | **QUYẾT ĐỊNH CHỐT:** Khi tỷ lệ số 0 giảm từ 82% xuống ~49–53% ở cấp tuần, mô hình `HistGB_L1` tối ưu trực tiếp sai số tuyệt đối đạt **WAPE = `0.6565` (< 1.0 rất xa)** và **$R^2 = 0.4302$** $\Rightarrow$ **Đủ tiêu chuẩn đưa vào vận hành lập kế hoạch nhập hàng hàng tuần (Weekly Replenishment)!** |

---

### FILE 9: `inventory_cost.py` — Đánh giá Chi phí Tồn kho Thực tế & Tối ưu Newsvendor

#### 1. File tương ứng
- **Đầu vào:** `train.parquet`, `val.parquet`, `test.parquet`, `test_predictions.parquet`, `intermittent_test_predictions.parquet`.
- **Hàm xử lý chính:** `calculate_inventory_cost(y_true, y_pred, c_u, c_o)`.
- **Đầu ra báo cáo & biểu đồ:**
  - `reports/inventory_cost_report.json`.
  - `reports/figures/inventory_cost_scenarios.png`, `reports/figures/service_level_tradeoff.png`.

#### 2. Thuật toán xử lý bên trong như thế nào?
1. **Hàm chi phí tồn kho bất đối xứng (Newsvendor Cost Function):**
   Với lượng bán thực tế $y_i$ và lượng hàng chuẩn bị theo dự báo $\hat{y}_i$:
   - Lượng thiếu hàng (Underage / Stockout): $u_i = \max(0, y_i - \hat{y}_i)$ (mỗi đơn vị thiếu bị phạt chi phí $C_u$).
   - Lượng tồn dư (Overage / Overstock): $o_i = \max(0, \hat{y}_i - y_i)$ (mỗi đơn vị thừa bị phạt chi phí lưu kho $C_o$).
   - **Tổng chi phí tồn kho:**
     $$\text{Total Cost} = \sum_{i=1}^N \left( C_u \cdot \max(0, y_i - \hat{y}_i) + C_o \cdot \max(0, \hat{y}_i - y_i) \right)$$
   - **3 chỉ số đo mức độ đáp ứng hàng hóa (Service Level):**
     - `service_level_pct`: Tỷ lệ ngày không bị thiếu hàng trên **toàn bộ các ngày** (kể cả ngày không có khách mua).
     - `positive_day_service_level_pct`: Tỷ lệ ngày không bị thiếu hàng **chỉ tính trên những ngày thực sự có khách mua ($y_i > 0$)**.
     - `volume_fill_rate_pct` (Tỷ lệ lấp đầy sản lượng): $\left(1 - \frac{\sum \max(0, y_i - \hat{y}_i)}{\sum y_i}\right) \times 100\%$.
2. **Thử nghiệm 2 cách tiếp cận Quantile Regression:**
   - **Cách 1 — Hồi quy phân vị vô điều kiện (`Quantile_0.75`, `Quantile_0.90`):** Huấn luyện `HistGradientBoostingRegressor(loss='quantile', quantile=q)` trên toàn bộ tập Train (chứa 78.9% số 0).
   - **Cách 2 — Mô hình 2 pha tối ưu chi phí `Hurdle_Newsvendor_Opt`:**
     - Huấn luyện bộ phân loại xác suất bán $p_i = \hat{P}(y_i > 0 \mid X_i)$ và các bộ hồi quy phân vị $q \in \{0.50, 0.75, 0.90\}$ **chỉ trên các ngày có bán ($y_i > 0$)**.
     - Với mỗi kịch bản $(C_u, C_o)$, tìm bộ tham số $(\text{quantile } q, \text{ngưỡng } \theta, \text{hệ số nhân } \text{scale})$ làm **cực tiểu hóa trực tiếp `Total Cost` trên tập Validation**, sau đó áp dụng lên tập Test.

#### 3. Các chỉ số thu được & Làm gì tiếp theo sau khi có kết quả?
| Mô hình (trong `inventory_cost_report.json`) | Chi phí Bán lẻ chuẩn ($C_u=3, C_o=1$) | Chi phí Khách VIP ($C_u=9, C_o=1$) | Chi phí Lưu kho đắt ($C_u=1, C_o=2$) | Positive-Day SL (%) | Volume Fill Rate (%) | Cách đọc chỉ số & Quyết định kinh doanh |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Quantile_0.75` (Vô điều kiện) | $10.02M (Tệ nhất) | $30.04M (Tệ nhất) | $3.35M | **0.02%** | **0.17%** | **Phát hiện bẫy lý thuyết:** Vì 78.9% ngày bằng 0, phân vị thứ 75% bằng 0! Mô hình dự báo gần như toàn số 0, làm hụt 99.83% lượng hàng khách cần mua $\Rightarrow$ **Tuyệt đối không dùng Quantile vô điều kiện với $\tau \le 0.80$ trên dữ liệu gián đoạn!** |
| `Lag1_Baseline` | $9.73M | $24.31M | $7.29M | 25.41% | 27.28% | Vừa tốn chi phí thiếu hàng vừa tốn chi phí tồn kho. |
| `TSB` | $8.43M | $21.12M | $6.30M | 34.88% | 36.78% | Cải thiện hơn Lag-1 nhờ làm mượt mức dự báo. |
| `TwoStageDemand` | $7.75M | $20.17M | $5.14M | 34.97% | 38.08% | Tốt cho mục tiêu cân bằng, nhưng chưa tự co giãn theo $(C_u, C_o)$. |
| **`Hurdle_Newsvendor_Opt`** | **$7.61M (Tốt nhất)** | **$17.14M** | **$3.18M (Tốt nhất)** | **36.13% – 65.36%** | **36.69% – 58.37%** | **Tự động thích nghi theo từng kịch bản chi phí:** Khi kho đắt ($C_o=2$) thì tự hạ tồn kho xuống ($3.18M); khi phạt thiếu hàng cao ($C_u=9$) thì tự nâng ngưỡng phân vị lên $q=0.75, \theta=0.25$ để đáp ứng **58.37%** sản lượng. |
| **`Quantile_0.90`** | $9.14M | **$16.58M (Tốt nhất)** | $12.09M | **71.60%** | **62.91%** | Vì $\tau = 0.90 > 0.80$ (vượt qua vùng 80% số 0), mô hình tạo đệm tồn kho an toàn cao, đáp ứng **62.91% tổng sản lượng** và tiết kiệm **7.73 triệu $** so với Baseline ở kịch bản VIP ($C_u=9, C_o=1$). |

---

## PHẦN 3: TỔNG KẾT "KỊCH BẢN TRẢ LỜI" KHI BẢO VỆ TRƯỚC HỘI ĐỒNG

Khi hội đồng hỏi câu tổng kết: **"Sau khi chạy hết 9 file và có toàn bộ chỉ số, kết luận cuối cùng của đồ án là gì?"**, bạn trả lời theo 4 ý gãy gọn sau:

1. **Về dữ liệu (`preprocess.py`, `eda.py`):** Dữ liệu bán lẻ thực tế có tính quy tụ Pareto rất cao (10% mã hàng chiếm 63.17% doanh thu) và tính gián đoạn cực đoan (**~80% quan sát ngày bằng 0**).
2. **Về khai phá không giám sát (`clustering.py`, `mining.py`):**
   - K-Means ($k=3$, Silhouette `0.5464`) giúp phân tách rõ 3 nhóm hàng (Phổ thông 77.9%, Bán chạy 20.7%, Chủ lực 1.4%).
   - Apriori tìm ra 932 luật kết hợp mạnh (Support $\ge 1\%$, Lift lên tới `45.97`) phục vụ bán chéo (Cross-selling).
   - Phát hiện bất thường chỉ ra 7.77% ngày bán có đơn sỉ đột biến, là cơ sở để chọn hàm mất mát bậc nhất **L1 (`absolute_error`)** thay vì bình phương sai số **L2**.
3. **Về dự báo nhu cầu (`modeling.py`, `intermittent_models.py`, `weekly_forecasting.py`):**
   - Ở **cấp độ Ngày**, mô hình 2 pha `TwoStageDemand` (Hurdle) vượt qua cả các mô hình 1 pha (Linear, RF, XGBoost) lẫn thuật toán chuyên dụng `Croston / SBA / TSB`, hạ WAPE từ `1.4543` xuống `1.0782`.
   - Tuy nhiên, để đưa vào vận hành thực tế, việc chuyển sang **cấp độ Tuần (`weekly_forecasting.py`)** mới là bước quyết định: giúp giảm tỷ lệ số 0 xuống ~50%, triệt tiêu sai số lệch ngày, nâng $R^2$ lên **`0.4524`** và hạ WAPE xuống **`0.6565`** (`HistGB_L1`).
4. **Về tối ưu hóa quyết định nhập kho (`inventory_cost.py`):**
   - Đồ án chứng minh không thể áp dụng máy móc Quantile Regression vô điều kiện ở các phân vị $\tau \le 0.80$ trên dữ liệu có 80% số 0 (vì bị sụp về 0), mà phải dùng **Hurdle Conditional Quantile (`Hurdle_Newsvendor_Opt`)** hoặc phân vị vượt ngưỡng số 0 (`Quantile_0.90`), giúp giảm hàng triệu đô-la chi phí đứt hàng cho doanh nghiệp.
