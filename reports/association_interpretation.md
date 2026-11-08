# Luật kết hợp Apriori — diễn giải

Tập gồm 39,536 hóa đơn; ngưỡng support ≥ 0.5%, confidence ≥ 15%. Có 5,187 luật ở ngưỡng khai phá và 932 luật có support ít nhất 1% được ưu tiên diễn giải.

## Các luật nổi bật theo lift

| Nếu hóa đơn có | Thì thường có | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| 22746 | 22745 | 1.17% | 81.74% | 45.97 |
| 22745 | 22746 | 1.17% | 65.58% | 45.97 |
| 22746 | 22748 | 1.20% | 84.04% | 44.78 |
| 22748 | 22746 | 1.20% | 63.88% | 44.78 |
| 22745 | 22748 | 1.41% | 79.23% | 42.22 |
| 22748 | 22745 | 1.41% | 75.07% | 42.22 |
| 22579 | 22578 | 1.11% | 79.02% | 40.68 |
| 22578 | 22579 | 1.11% | 56.90% | 40.68 |
| 84596F | 84596B | 1.16% | 74.43% | 40.15 |
| 84596B | 84596F | 1.16% | 62.35% | 40.15 |

## Cách đọc

Support là tỷ lệ hóa đơn chứa cả hai phía của luật. Confidence là xác suất thấy sản phẩm bên phải trong các hóa đơn có sản phẩm bên trái. Lift so sánh confidence với tỷ lệ nền của sản phẩm bên phải; lift lớn hơn 1 cho thấy hai sản phẩm đồng xuất hiện thường hơn mức ngẫu nhiên độc lập. Đây là quan hệ đồng mua, không chứng minh một sản phẩm gây ra việc mua sản phẩm kia.

## Lưu ý phương pháp

Các luật được xếp hạng cần được kiểm tra cùng support, vì lift có thể rất cao khi sự kiện hiếm. Kết quả phục vụ khám phá giỏ hàng; trước khi dùng cho gợi ý bán hàng cần đánh giá ổn định theo thời gian và kiểm tra mô tả sản phẩm.
