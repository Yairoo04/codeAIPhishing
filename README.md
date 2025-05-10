
# Phát hiện Phishing Dựa trên AI

## Tổng Quan
Kho lưu trữ này chứa mã nguồn cho một hệ thống phát hiện phishing dựa trên AI, phát hiện các mối đe dọa phishing qua các dạng: **URL**, **Email**, **Tệp PDF** và **Hình ảnh**. Dự án này áp dụng các kỹ thuật học máy, cụ thể là **Random Forest (RF)** và **Mạng Nơ-ron Tích chập (CNN)**, để phân loại chính xác các nội dung phishing.

Hệ thống được thiết kế để phát hiện và phân loại các cuộc tấn công phishing qua nhiều kênh khác nhau, cung cấp một giải pháp tích hợp nhằm nâng cao an ninh mạng. Các mô hình đã được huấn luyện trên các bộ dữ liệu công khai từ các nguồn như **PhishTank**, **OpenPhish**, và **CIRCL**.

## Tính Năng
- **Phát hiện Phishing URL**: Phân loại các URL là hợp lệ hoặc phishing bằng mô hình Random Forest được huấn luyện từ các đặc trưng của URL.
- **Phát hiện Phishing Email**: Phát hiện các email phishing dựa trên các đặc trưng trích xuất từ tiêu đề và nội dung email sử dụng mô hình Random Forest.
- **Phát hiện Phishing PDF**: Phân loại các tệp PDF là hợp lệ hoặc phishing dựa trên các metadata và đặc trưng nhúng.
- **Phát hiện Phishing Hình ảnh**: Phát hiện phishing qua hình ảnh (bao gồm mã QR) bằng sự kết hợp của mô hình CNN và Random Forest.

## Bộ Dữ Liệu
Các bộ dữ liệu được sử dụng trong dự án này bao gồm:
- **Dữ liệu Phishing URL**: Thu thập từ [PhishTank](https://www.phishtank.com) và các nguồn khác như Majestic và OpenPhish.
- **Dữ liệu Phishing Email**: Lấy từ SpamAssassin Public Corpus chứa các email phishing.
- **Dữ liệu Phishing PDF**: Thu thập từ [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/pdfmal-2022.html).
- **Dữ liệu Phishing Hình ảnh**: Tải từ [CIRCL Phishing Dataset](https://www.circl.lu/opendata/datasets/circl-phishing-dataset-01/).

## Yêu Cầu
Để chạy mã nguồn, bạn cần cài đặt các thư viện sau:
- Python 3.x
- `scikit-learn`
- `pandas`
- `numpy`
- `tensorflow` (cho CNN)
- `Flask` (cho giao diện web)
- `matplotlib` (cho các hình ảnh)
- `pyzbar` (cho phát hiện mã QR)

Cài đặt các thư viện cần thiết bằng lệnh:
```
pip install -r requirements.txt
```

## Cấu Trúc Dự Án
- **/app**: Chứa các mô hình học máy cho phát hiện phishing qua email, tệp, hình ảnh và URL. Các mô hình được tổ chức thành các thư mục riêng biệt như `Phishing_Email_Models`, `Phishing_File_Models`, `Phishing_Image_Models`, và `Phishing_URL_Models`. Các tệp liên quan đến huấn luyện và tiền xử lý dữ liệu như `train_RF_email.py`, `train_random_forest.py` cho từng loại dữ liệu đều được đặt trong các thư mục tương ứng.
- **/static**: Chứa các tệp JS và CSS cần thiết cho giao diện người dùng.
- **/templates**: Chứa các tệp HTML và các tài nguyên tĩnh khác như ảnh `confusion_matrix.png` và `training_history.png`.
- **/dataset_Email**, **/dataset_File**, **/dataset_Image**, **/dataset_URL**: Các thư mục này chứa dữ liệu đã thu thập và tiền xử lý cho các loại phishing khác nhau.

## Chạy Ứng Dụng

### Giao Diện Web:
1. Clone kho lưu trữ:
   ```bash
   git clone https://github.com/Yairoo04/codeAIPhishing.git
   cd codeAIPhishing
   ```

2. Cài đặt các phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

3. Di chuyển vào thư mục `app` và chạy ứng dụng web:
   ```bash
   python app/main.py
   ```
   Ứng dụng web sẽ chạy trên `http://localhost:5000`.

### Các Mô Hình Phishing:
- **Phishing Email**: Mô hình được huấn luyện trong thư mục `Phishing_Email_Models` sử dụng tệp `train_RF_email.py`.
- **Phishing File (PDF)**: Mô hình huấn luyện được thực hiện trong thư mục `Phishing_File_Models` với tệp `train_random_forest.py`.
- **Phishing URL**: Mô hình huấn luyện cho phát hiện phishing qua URL nằm trong thư mục `Phishing_URL_Models`, với tệp `rf_url_model.py`.
- **Phishing Image**: Mô hình sử dụng CNN trong thư mục `Phishing_Image_Models`.

### Giao Diện Chrome Extension:
1. Trong thư mục `static/`, tải tệp `app.js` và `styles.css` vào Chrome Extension để kiểm tra phishing URL trực tiếp khi duyệt web.

## Huấn Luyện lại Mô Hình

Nếu bạn muốn huấn luyện lại các mô hình, bạn cần chạy các tệp huấn luyện trong các thư mục tương ứng. Dưới đây là hướng dẫn chi tiết để huấn luyện lại mô hình cho từng loại dữ liệu:

#### 1. **Huấn luyện lại mô hình Phishing URL**:
- Di chuyển vào thư mục `Phishing_URL_Models` và chạy tệp `rf_url_model.py`:
   ```bash
   cd app/Phishing_URL_Models
   python rf_url_model.py
   ```

#### 2. **Huấn luyện lại mô hình Phishing Email**:
- Di chuyển vào thư mục `Phishing_Email_Models` và chạy tệp `train_RF_email.py`:
   ```bash
   cd app/Phishing_Email_Models
   python train_RF_email.py
   ```

#### 3. **Huấn luyện lại mô hình Phishing PDF**:
- Di chuyển vào thư mục `Phishing_File_Models` và chạy tệp `train_random_forest.py`:
   ```bash
   cd app/Phishing_File_Models
   python train_random_forest.py
   ```

#### 4. **Huấn luyện lại mô hình Phishing Hình ảnh**:
- Di chuyển vào thư mục `Phishing_Image_Models` và chạy tệp `train_cnn_image.py`:
   ```bash
   cd app/Phishing_Image_Models
   python train_cnn_image.py
   ```

Sau khi các mô hình được huấn luyện lại, bạn có thể kiểm tra lại chúng bằng cách sử dụng các tập dữ liệu mới và đánh giá mô hình như đã hướng dẫn trong phần **Đánh giá mô hình**.

## Đánh Giá Mô Hình
Các mô hình được đánh giá bằng các chỉ số sau:
- **Accuracy**: Tỷ lệ phân loại đúng trên tổng số mẫu.
- **Precision**: Tỷ lệ phân loại đúng các mẫu "Phishing".
- **Recall**: Tỷ lệ các mẫu "Phishing" dự đoán đúng trên tổng số mẫu thực tế là "Phishing".

Kết quả ví dụ của mô hình Random Forest trên các loại dữ liệu phishing:

| Mô Hình            | Accuracy | Precision | Recall |
|--------------------|----------|-----------|--------|
| **Phishing URL**    | 93.0%    | 95.0%     | 95.0%  |
| **Phishing PDF**    | 99.0%    | 99.0%     | 99.0%  |
| **Phishing Email**  | 99.0%    | 100%      | 99.0%  |
| **Phishing Hình ảnh** | 88.0%    | 85.0%     | 87.0%  |

## Kết Luận
Hệ thống này chứng minh sức mạnh của các kỹ thuật học máy, đặc biệt là **Random Forest** và **CNN**, trong việc phát hiện các mối đe dọa phishing qua nhiều dạng dữ liệu khác nhau như URL, email, PDF và hình ảnh. Dự án hướng đến việc cải thiện an ninh mạng bằng cách cung cấp một giải pháp hiệu quả và dễ dàng triển khai để phát hiện phishing.

## Cộng Tác
Hãy thoải mái tạo nhánh mới từ kho lưu trữ này, đóng góp các cải tiến hoặc đề xuất các tính năng mới. Vui lòng đảm bảo tuân theo hướng dẫn mã nguồn của dự án và cung cấp đầy đủ tài liệu cho các thay đổi.

## Giấy Phép
Dự án này được cấp phép dưới Giấy phép MIT - xem chi tiết trong tệp [LICENSE](LICENSE).
