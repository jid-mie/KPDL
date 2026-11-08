# **I. MÔ TẢ BÀI TOÁN**

## **1\. Lý do chọn đề tài**

Trong những năm gần đây, sự phát triển mạnh mẽ của thương mại điện tử và hoạt động bán lẻ đã tạo ra một lượng lớn dữ liệu giao dịch. Mỗi giao dịch bán hàng có thể chứa nhiều thông tin như mã sản phẩm, số lượng sản phẩm được mua, thời gian giao dịch, đơn giá, khách hàng và quốc gia. Những dữ liệu này không chỉ phản ánh hoạt động kinh doanh đã xảy ra mà còn có thể được khai thác để tìm ra các xu hướng, quy luật và đặc điểm trong hành vi tiêu dùng.

Trong hoạt động kinh doanh bán lẻ, việc xác định nhu cầu tiêu thụ sản phẩm trong tương lai có ý nghĩa quan trọng đối với quá trình lập kế hoạch kinh doanh và quản lý hàng hóa. Nếu doanh nghiệp dự trữ quá nhiều sản phẩm, chi phí lưu kho có thể tăng và có nguy cơ phát sinh hàng tồn. Ngược lại, nếu lượng hàng dự trữ không đáp ứng được nhu cầu, doanh nghiệp có thể bỏ lỡ cơ hội bán hàng và ảnh hưởng đến khả năng phục vụ khách hàng. Do đó, việc phân tích dữ liệu bán hàng trong quá khứ để nhận diện xu hướng và dự đoán lượng sản phẩm có khả năng được tiêu thụ trong tương lai là một bài toán có tính ứng dụng thực tế.

Tuy nhiên, dữ liệu giao dịch bán lẻ thường có kích thước lớn, chứa nhiều giá trị thiếu, giao dịch bị hủy, dữ liệu không hợp lệ và các biến có sự khác biệt về kiểu dữ liệu. Vì vậy, trước khi xây dựng mô hình dự đoán, cần thực hiện các bước tiền xử lý, làm sạch và biến đổi dữ liệu phù hợp. Đây cũng là những nội dung quan trọng của khai phá dữ liệu.

Trong đề tài này, nhóm sử dụng bộ dữ liệu **Online Retail II** được cung cấp bởi **UCI Machine Learning Repository**. Bộ dữ liệu chứa các giao dịch bán lẻ thực tế với hơn một triệu bản ghi, bao gồm các thông tin về hóa đơn, sản phẩm, số lượng, thời gian giao dịch, đơn giá, khách hàng và quốc gia. Dữ liệu này tạo điều kiện để thực hiện nhiều bước khai phá dữ liệu như phân tích thống kê, phát hiện dữ liệu bất thường, phân tích xu hướng tiêu thụ sản phẩm và xây dựng các đặc trưng phục vụ dự đoán.

Từ dữ liệu giao dịch ban đầu, đề tài tiến hành tổng hợp số lượng sản phẩm được bán theo thời gian để xây dựng biến **nhu cầu tiêu thụ sản phẩm (Product Demand)**. Trên cơ sở đó, các đặc trưng về thời gian, lịch sử bán hàng và giá sản phẩm được xây dựng nhằm phục vụ cho bài toán dự đoán nhu cầu trong tương lai.

Bên cạnh bài toán dự đoán, đề tài còn áp dụng một số kỹ thuật khai phá dữ liệu nhằm tìm hiểu sâu hơn về dữ liệu bán lẻ, chẳng hạn như phân nhóm sản phẩm theo đặc điểm tiêu thụ, phát hiện các giao dịch hoặc thời điểm có nhu cầu bất thường và tìm kiếm mối quan hệ giữa các sản phẩm thường được mua cùng nhau.

Thông qua đề tài, nhóm hướng tới việc xây dựng một quy trình khai phá dữ liệu hoàn chỉnh từ **thu thập dữ liệu, tiền xử lý, phân tích, khai phá dữ liệu, xây dựng đặc trưng, huấn luyện mô hình đến đánh giá kết quả dự đoán**. Kết quả của đề tài có thể được sử dụng như một cơ sở hỗ trợ doanh nghiệp phân tích hoạt động bán hàng và lập kế hoạch nhu cầu sản phẩm trong tương lai.

---

## **2\. Tổng quan bài toán**

Bài toán của đề tài được xây dựng dựa trên dữ liệu lịch sử giao dịch bán lẻ. Mỗi bản ghi trong dữ liệu thể hiện một dòng sản phẩm trong một hóa đơn, bao gồm mã sản phẩm, số lượng mua, thời gian giao dịch, đơn giá, khách hàng và quốc gia.

Dữ liệu ban đầu có dạng:

| InvoiceNo | StockCode | Description | Quantity | InvoiceDate | UnitPrice | CustomerID | Country |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 536365 | 85123A | Product A | 6 | 01/12/2010 08:26 | 2.55 | 17850 | United Kingdom |
| 536365 | 71053 | Product B | 6 | 01/12/2010 08:26 | 3.39 | 17850 | United Kingdom |
| 536366 | 22633 | Product C | 10 | 01/12/2010 08:28 | 1.85 | 17850 | United Kingdom |

Tuy nhiên, dữ liệu giao dịch ở dạng trên chưa thể sử dụng trực tiếp cho bài toán dự đoán nhu cầu. Vì vậy, dữ liệu cần được làm sạch và tổng hợp theo **sản phẩm và thời gian**.

Ví dụ:

Dữ liệu giao dịch  
        ↓  
Loại bỏ dữ liệu không hợp lệ  
        ↓  
Xử lý giao dịch hủy  
        ↓  
Xử lý giá trị thiếu  
        ↓  
Tổng hợp theo sản phẩm \+ ngày  
        ↓  
Product Daily Demand  
        ↓  
Xây dựng đặc trưng  
        ↓  
Huấn luyện mô hình  
        ↓  
Dự đoán nhu cầu

Trong đó, nhu cầu tiêu thụ của một sản phẩm trong ngày được xác định dựa trên tổng số lượng sản phẩm được bán trong ngày đó:

> **Daily Demand \= Tổng Quantity của sản phẩm trong một ngày**

Ví dụ:

| Ngày | Sản phẩm | Tổng số lượng bán |
| ----- | ----- | ----- |
| 01/01 | A | 25 |
| 02/01 | A | 31 |
| 03/01 | A | 28 |
| 04/01 | A | 35 |

Từ chuỗi dữ liệu lịch sử này, mô hình sẽ học được các đặc điểm về xu hướng tiêu thụ và sử dụng chúng để dự đoán nhu cầu trong tương lai.

---

## **3\. Quy trình thực hiện**

Quy trình thực hiện đề tài được xây dựng theo các bước chính:

       Dataset Online Retail II  
                  ↓  
          Khảo sát dữ liệu  
                  ↓  
          Tiền xử lý dữ liệu  
                  ↓  
         Phân tích dữ liệu  
                  ↓  
        Khai phá dữ liệu  
          ↙       ↓       ↘  
   Clustering  Association  Anomaly  
                  ↓  
        Feature Engineering  
                  ↓  
        Xây dựng tập dữ liệu  
                  ↓  
        Chia Train / Validation / Test  
                  ↓  
          Xây dựng mô hình  
                  ↓  
     ┌────────────┼────────────┐  
     ↓            ↓            ↓  
 Linear Reg. Random Forest   XGBoost  
     └────────────┼────────────┘  
                  ↓  
          Đánh giá mô hình  
                  ↓  
       Dự đoán nhu cầu sản phẩm  
                  ↓  
            Trực quan hóa

Quy trình trên đảm bảo đề tài không chỉ tập trung vào xây dựng mô hình dự đoán mà còn thể hiện đầy đủ các bước của một bài toán khai phá dữ liệu.

---

## **4\. Phân tích dữ liệu thô**

Bộ dữ liệu **Online Retail II** chứa thông tin giao dịch bán lẻ trong khoảng thời gian từ **01/12/2009 đến 09/12/2011**. Các trường dữ liệu chính bao gồm mã hóa đơn, mã sản phẩm, tên sản phẩm, số lượng, thời gian giao dịch, đơn giá, mã khách hàng và quốc gia.

Một số trường dữ liệu có vai trò quan trọng đối với đề tài:

| Thuộc tính | Ý nghĩa | Vai trò |
| ----- | ----- | ----- |
| `InvoiceNo` | Mã hóa đơn | Xác định giao dịch |
| `StockCode` | Mã sản phẩm | Xác định sản phẩm |
| `Description` | Tên sản phẩm | Mô tả sản phẩm |
| `Quantity` | Số lượng sản phẩm | Biến quan trọng để xác định nhu cầu |
| `InvoiceDate` | Thời gian giao dịch | Xây dựng chuỗi thời gian |
| `UnitPrice` | Đơn giá sản phẩm | Đặc trưng dự đoán |
| `CustomerID` | Mã khách hàng | Phân tích khách hàng |
| `Country` | Quốc gia | Phân tích theo khu vực |

Trong quá trình khảo sát dữ liệu thô, cần kiểm tra các vấn đề như:

* Giá trị bị thiếu.  
* Dữ liệu trùng lặp.  
* Số lượng sản phẩm có giá trị bất thường.  
* Đơn giá bằng 0 hoặc không hợp lệ.  
* Các hóa đơn bị hủy.  
* Định dạng thời gian.  
* Các sản phẩm có số lượng giao dịch quá thấp.

Đặc biệt, các hóa đơn có mã bắt đầu bằng ký tự **“C”** được UCI mô tả là các giao dịch bị hủy. Những trường hợp này cần được xem xét và xử lý trước khi tính toán nhu cầu tiêu thụ, nhằm tránh làm sai lệch số lượng sản phẩm bán ra.

---

### **II. QUY TRÌNH KHAI PHÁ DỮ LIỆU**

#### **1\. Tiền xử lý dữ liệu**

Trước khi phân tích, dữ liệu thô cần được làm sạch. Bộ dữ liệu Online Retail II có hơn một triệu bản ghi nhưng chứa nhiều vấn đề như giá trị bị thiếu, số lượng âm, đơn giá bằng 0, hóa đơn bị hủy và dữ liệu trùng lặp. Nếu bỏ qua các bước này, các số liệu thống kê và mô hình dự đoán sau đó sẽ bị sai lệch.

**1.1.1. Loại bỏ thuộc tính dư thừa**

Dữ liệu ban đầu gồm 8 thuộc tính. Trong ba nhánh phân tích (K-Means, Apriori, dự đoán nhu cầu), mỗi nhánh chỉ dùng đến một nhóm thuộc tính, nên không có cột nào bị loại bỏ hoàn toàn. Tuy nhiên, một số cột không đóng góp vào việc xác định nhu cầu và được loại khỏi tập dữ liệu huấn luyện:

| Thuộc tính | Quyết định | Lý do |
| ----- | ----- | ----- |
| Description | Giữ cho EDA, loại khỏi mô hình | Không có ý nghĩa số liệu, dễ gây nhiễu do khác nhau về chính tả cho cùng một sản phẩm |
| CustomerID | Không dùng cho dự đoán nhu cầu | Thiếu khoảng 25% giá trị; nhu cầu theo sản phẩm không phụ thuộc vào một khách hàng cụ thể |
| Country | Giữ lại khi phân tích theo khu vực | Tập trung chủ yếu vào Vương quốc Anh nên không đủ biến thể để làm đặc trưng cho mô hình |

**1.1.2. Loại bỏ bản ghi nhiễu**

Các bản ghi nhiễu trong bộ dữ liệu được xác định theo bốn tiêu chí:

Thứ nhất, loại bỏ các hóa đơn có mã bắt đầu bằng ký tự "C". Đây là các giao dịch bị hủy và thường đi kèm số lượng âm. Không xử lý chúng thì tổng số lượng bán của một sản phẩm sẽ bị âm ở một số ngày.

Thứ hai, loại bỏ các bản ghi có số lượng âm nhưng không thuộc hóa đơn hủy. Những trường hợp này phần lớn là lỗi ghi dữ liệu.

Thứ ba, loại bỏ bản ghi có UnitPrice nhỏ hơn hoặc bằng 0\. Đây là giá trị khuyến mãi hoặc lỗi nhập liệu, không đại diện cho giá thực của sản phẩm.

Thứ tư, loại bỏ các bản ghi trùng lặp hoàn toàn về cả 8 thuộc tính.

Trong quá trình xử lý, cần ghi lại số lượng bản ghi bị loại ở mỗi bước để minh bạch cho kết quả:

| Bước xử lý | Số bản ghi bị loại | Tỷ lệ (%) |
| ----- | ----- | ----- |
| Hóa đơn hủy (InvoiceNo bắt đầu bằng "C") | \[điền từ kết quả chạy\] | \[ \] |
| Số lượng âm ngoài hóa đơn hủy | \[ \] | \[ \] |
| Đơn giá ≤ 0 | \[ \] | \[ \] |
| Bản ghi trùng lặp hoàn toàn | \[ \] | \[ \] |
| **Tổng cộng** | **\[ \]** | **\[ \]** |

**1.1.3. Xử lý dữ liệu bị thiếu**

Vấn đề thiếu dữ liệu tập trung ở hai cột: CustomerID và Description. Mức độ thiếu của từng trường cần được đo trước khi quyết định cách xử lý:

| Thuộc tính | Số giá trị thiếu | Tỷ lệ (%) | Cách xử lý |
| ----- | ----- | ----- | ----- |
| CustomerID | \[ \] | \[ \] | Loại khỏi tập huấn luyện vì không dùng trong dự đoán nhu cầu |
| Description | \[ \] | \[ \] | Xác định lại bằng StockCode; nếu không tìm được thì loại |
| UnitPrice | \[ \] | \[ \] | Không phát sinh thêm giá trị thiếu sau bước lọc |
| Quantity | \[ \] | \[ \] | Không phát sinh thêm giá trị thiếu sau bước lọc |

Đối với Description, vì một mã sản phẩm thường có mô tả thống nhất, nên giá trị thiếu có thể được điền bằng giá trị xuất hiện nhiều nhất của mã sản phẩm đó. Nếu mã sản phẩm cũng không có mô tả nào thì loại bản ghi.

Không nên điền giá trị thiếu cho CustomerID bằng giá trị mặc định (ví dụ "Unknown") vì trong trường hợp này đây không phải là biến quan trọng. Điền giá trị sai sẽ tạo ra ảo giác về việc dữ liệu đã đầy đủ.

**1.1.4. Xử lý dữ liệu không nhất quán**

Có ba vấn đề không nhất quán cần xử lý. Đầu tiên là trường StockCode, có mã gồm chữ và số (ví dụ "85123A"), có mã chỉ gồm chữ (chỉ số như "POST", "BANK CHARGES", "DOT"), đây là mã của các khoản phí hoặc dịch vụ, không phải sản phẩm bán ra, nên cần được loại khỏi tập dữ liệu phục vụ dự đoán nhu cầu.

Thứ hai, cùng một sản phẩm có thể có mô tả khác nhau về chính tả hoặc khoảng trắng. Giải pháp là làm sạch mô tả (bỏ khoảng trắng thừa, viết hoa đồng nhất) rồi kiểm tra số lượng mô tả còn lại cho mỗi StockCode.

Thứ ba, cột InvoiceDate ở định dạng chuỗi. Cần chuyển sang kiểu datetime theo định dạng ngày/giờ, sau đó trích xuất thêm các trường: Năm, Tháng, Ngày, Thứ trong tuần, Tuần trong năm và Đánh dấu ngày lễ (nếu có). Việc xử lý định dạng thời gian không chỉ làm sạch dữ liệu mà còn tạo ra các đặc trưng cho mô hình.

**1.2. Tích hợp dữ liệu**

Vì mỗi bản ghi chỉ đại diện cho một dòng sản phẩm trong một hóa đơn, nên dữ liệu cần được tổng hợp lại theo sản phẩm và thời gian. Trước khi tổng hợp, trường InvoiceDate được rút về giá trị ngày (bỏ phần giờ). Sau đó, tính tổng Quantity của mỗi StockCode trong mỗi ngày.

Kết quả là bảng dữ liệu theo mẫu:

| StockCode | Date | DailyDemand | UnitPrice (TB ngày) | Số hóa đơn trong ngày |
| ----- | ----- | ----- | ----- | ----- |
| 85123A | 01/12/2009 | 120 | 2.55 | 4 |
| 85123A | 02/12/2009 | 98 | 2.55 | 3 |
| 71053 | 01/12/2009 | 85 | 3.39 | 2 |
| 71053 | 02/12/2009 | 90 | 3.39 | 2 |

Một điểm cần lưu ý ở bước tổng hợp: nếu một sản phẩm không được bán trong ngày nào đó, điều đó không có nghĩa là giá trị là NaN hay bị thiếu, mà DailyDemand thực sự bằng 0\. Vì vậy cần tạo bảng bao gồm toàn bộ các ngày trong khoảng thời gian của dữ liệu cho mỗi sản phẩm, sau đó điền 0 cho các ngày không có giao dịch. Nếu bỏ qua bước này, các ngày không có giao dịch sẽ bị coi là ngày bị thiếu và chuỗi thời gian sẽ bị sai lệch nghiêm trọng.

**1.3. Biến đổi dữ liệu**

Bước biến đổi có hai mục đích: tạo biến đích (target) và tạo đặc trưng (features).

Về biến đích, DailyDemand là biến cần dự đoán. Quan sát trước tiên là phân phối của nó. Số lượng bán trong một ngày có phân phối lệch phải mạnh, có nhiều giá trị rất lớn so với trung vị. Ba biến đổi thường dùng bao gồm biến đổi logarit, biến đổi căn bậc hai và chuyển sang dạng phân loại (nhóm bán thấp/cao). Việc chọn biến đổi nào sẽ được minh họa ở chương trình thực nghiệm.

Về đặc trưng, nhóm xây dựng các biến sau cho mỗi cặp (StockCode, ngày):

| Nhóm đặc trưng | Đặc trưng | Ý nghĩa |
| ----- | ----- | ----- |
| Đặc trưng thời gian | Tháng, thứ trong tuần, tuần, cờ ngày lễ | Mô tả yếu tố mùa vụ |
| Đặc trưng lịch sử | Trung bình 7 ngày trước, 14 ngày trước, 28 ngày trước; tổng 7 ngày trước; độ lệch chuẩn 28 ngày; ngày không bán gần nhất | Mô tả xu hướng và tính ổn định |
| Đặc trưng giá | UnitPrice trung bình của sản phẩm; thay đổi giá so với ngày trước | Mô tả ảnh hưởng của giá |
| Đặc trưng sản phẩm | Tổng số ngày có giao dịch; tổng số lượng bán trong toàn bộ dữ liệu; nhóm sản phẩm (từ K-Means) | Mô tả quy mô và loại sản phẩm |

Tất cả các đặc trưng lịch sử phải được tính từ dữ liệu trước ngày đang dự đoán, không được sử dụng giá trị của chính ngày đó. Đây là điều kiện bắt buộc để tránh rò rỉ dữ liệu.

#### **2\. Phân tích dữ liệu**

Sau khi làm sạch, dữ liệu được phân tích theo bốn hướng. Đầu tiên là thống kê mô tả: số dòng, số sản phẩm, số khách hàng, khoảng thời gian dữ liệu, phân vị số lượng bán và đơn giá. Thứ hai là phân tích theo thời gian: tổng số lượng bán theo tuần và theo tháng để nhận diện mùa vụ, trong đó cần chú ý vào cao điểm cuối năm.

Thứ ba là phân tích theo sản phẩm: số lượng giao dịch, tổng số lượng bán và doanh thu của từng StockCode, từ đó xác định phân bố tập trung (tỷ lệ đóng góp của nhóm sản phẩm bán chạy nhất vào tổng doanh thu).

Thứ tư là phân tích theo khách hàng và quốc gia: số lượng giao dịch và chi tiêu trung bình theo khách hàng, phân bố khách hàng theo quốc gia. Phân tích này không trực tiếp phục vụ dự đoán nhưng cho thấy đặc điểm của tập dữ liệu.

Mọi số liệu phân tích cần được trình bày dưới dạng bảng hoặc hình, kèm số liệu cụ thể.

#### **3\. Khai phá dữ liệu**

Tại chương này, nhóm sử dụng ba kỹ thuật khai phá dữ liệu. Cụ thể:

Phân nhóm sản phẩm bằng K-Means áp dụng trên các đặc trưng tổng hợp theo sản phẩm (trung bình số lượng bán, tần suất giao dịch, mức giá trung bình, độ biến động). Số nhóm được chọn bằng phương pháp Elbow và Silhouette.

Khai phá luật kết hợp bằng Apriori áp dụng trên từng hóa đơn, coi mỗi hóa đơn là một tập mục và mỗi StockCode là một mặt hàng. Tham số ngưỡng hỗ trợ và tin cậy sẽ được trình bày cụ thể ở chương trình thực nghiệm.

Phát hiện dữ liệu bất thường áp dụng trên tổng số lượng bán theo ngày và trên tổng doanh thu theo ngày, nhằm nhận diện các thời điểm có nhu cầu bất thường so với thông thường của chuỗi.

Ba kỹ thuật này được trình bày chi tiết ở Chương III và kết quả thực nghiệm ở Chương IV.

#### **4\. Phân chia tập dữ liệu**

Vì biến cần dự đoán là nhu cầu theo thời gian, nên tập dữ liệu được chia theo thứ tự thời gian, không chia ngẫu nhiên. Nếu chia ngẫu nhiên, dữ liệu của ngày hôm sau có thể nằm trong tập huấn luyện và làm cho việc huấn luyện "nhìn thấy" kết quả tương lai.

Mốc chia được xác định như sau:

| Tập | Khoảng thời gian | Mục đích |
| ----- | ----- | ----- |
| Train | 01/12/2009 → 30/11/2010 | Huấn luyện mô hình |
| Validation | 01/12/2010 → 28/02/2011 | Chọn siêu tham số, so sánh mô hình |
| Test | 01/03/2011 → 09/12/2011 | Đánh giá cuối cùng |

Sau khi chia, kiểm tra phân bố DailyDemand ở ba tập để đảm bảo không có sự lệch quá lớn giữa tập huấn luyện và tập kiểm tra, từ đó có cơ sở giải thích cho kết quả đánh giá sau này.

