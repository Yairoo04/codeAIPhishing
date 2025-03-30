# README

## Danh sách tệp dữ liệu

### 1. Benign_list_big_final.csv
Tệp này chứa danh sách các URL hợp lệ với tổng số 35.300. Dữ liệu được lấy từ Đại học New Brunswick.
Nguồn: [CIC Dataset - URL 2016](https://www.unb.ca/cic/datasets/url-2016.html).

### 2. online-valid.csv
Tệp này được tải xuống từ dịch vụ mã nguồn mở PhishTank. Dịch vụ này cung cấp danh sách các URL lừa đảo ở nhiều định dạng như CSV, JSON,... và được cập nhật hàng giờ.
Để tải xuống dữ liệu mới nhất, truy cập: [PhishTank Developer Info](https://www.phishtank.com/developer_info.php).

### 3. data.csv
Tệp này là tập hợp từ hai tệp trên, bao gồm cả URL lừa đảo (phishing) và URL hợp lệ (legitimate).

#### - Nguồn danh sách URL lừa đảo (Phishing URLs):
  - **Tên nguồn:** OpenPhish
  - **URL:** [OpenPhish](https://openphish.com/)
  - **Mô tả:** OpenPhish cung cấp danh sách các URL lừa đảo được cập nhật liên tục dựa trên việc thu thập dữ liệu từ nhiều nguồn khác nhau. Danh sách có thể được truy cập công khai tại: [OpenPhish Feed](https://openphish.com/feed.txt).

#### - Nguồn danh sách URL hợp lệ (Legitimate URLs):
  - **Tên nguồn:** Majestic Million
  - **URL:** [Majestic Million](https://majestic.com/reports/majestic-million)
  - **Mô tả:** Majestic Million là danh sách 1 triệu tên miền phổ biến nhất trên Internet, được thu thập và xếp hạng dựa trên số lượng backlink. Dữ liệu có thể được tải về dưới dạng CSV tại: [Majestic Million CSV](https://downloads.majestic.com/majestic_million.csv).

## Xử lý dữ liệu hình ảnh
Phần dữ liệu ảnh được xử lý để chụp ảnh từ các URL thu thập được nhằm phục vụ nghiên cứu và phân tích.
