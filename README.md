Phát hiện Phishing Dựa trên AI
Tổng Quan
Kho lưu trữ này chứa mã nguồn cho một hệ thống phát hiện phishing dựa trên AI, phát hiện các mối đe dọa phishing qua các dạng: URL, Email, Tệp PDF và Hình ảnh. Dự án này áp dụng các kỹ thuật học máy, cụ thể là Random Forest (RF) và Mạng Nơ-ron Tích chập (CNN), để phân loại chính xác các nội dung phishing.

Hệ thống được thiết kế để phát hiện và phân loại các cuộc tấn công phishing qua nhiều kênh khác nhau, cung cấp một giải pháp tích hợp nhằm nâng cao an ninh mạng. Các mô hình đã được huấn luyện trên các bộ dữ liệu công khai từ các nguồn như PhishTank, OpenPhish, và CIRCL.

Tính Năng
Phát hiện Phishing URL: Phân loại các URL là hợp lệ hoặc phishing bằng mô hình Random Forest được huấn luyện từ các đặc trưng của URL.

Phát hiện Phishing Email: Phát hiện các email phishing dựa trên các đặc trưng trích xuất từ tiêu đề và nội dung email sử dụng mô hình Random Forest.

Phát hiện Phishing PDF: Phân loại các tệp PDF là hợp lệ hoặc phishing dựa trên các metadata và đặc trưng nhúng.

Phát hiện Phishing Hình ảnh: Phát hiện phishing qua hình ảnh (bao gồm mã QR) bằng sự kết hợp của mô hình CNN và Random Forest.

Bộ Dữ Liệu
Các bộ dữ liệu được sử dụng trong dự án này bao gồm:

Dữ liệu Phishing URL: Thu thập từ PhishTank và các nguồn khác như Majestic và OpenPhish.

Dữ liệu Phishing Email: Lấy từ SpamAssassin Public Corpus chứa các email phishing.

Dữ liệu Phishing PDF: Thu thập từ Canadian Institute for Cybersecurity.

Dữ liệu Phishing Hình ảnh: Tải từ CIRCL Phishing Dataset.

Yêu Cầu
Để chạy mã nguồn, bạn cần cài đặt các thư viện sau:

Python 3.x

scikit-learn

pandas

numpy

tensorflow (cho CNN)

Flask (cho giao diện web)

matplotlib (cho các hình ảnh)

pyzbar (cho phát hiện mã QR)

Cài đặt các thư viện cần thiết bằng lệnh:

nginx
Sao chép
pip install -r requirements.txt
Cấu Trúc Dự Án
/app: Chứa các mô hình học máy cho phát hiện phishing qua email, tệp, hình ảnh và URL. Các mô hình được tổ chức thành các thư mục riêng biệt như Phishing_Email_Models, Phishing_File_Models, Phishing_Image_Models, và Phishing_URL_Models. Các tệp liên quan đến huấn luyện và tiền xử lý dữ liệu như train_RF_email.py, train_random_forest.py cho từng loại dữ liệu đều được đặt trong các thư mục tương ứng.

/static: Chứa các tệp JS và CSS cần thiết cho giao diện người dùng.

/templates: Chứa các tệp HTML và các tài nguyên tĩnh khác như ảnh confusion_matrix.png và training_history.png.

/dataset_Email, /dataset_File, /dataset_Image, /dataset_URL: Các thư mục này chứa dữ liệu đã thu thập và tiền xử lý cho các loại phishing khác nhau.

Chạy Ứng Dụng
Giao Diện Web:
Clone kho lưu trữ:

bash
Sao chép
git clone https://github.com/Yairoo04/codeAIPhishing.git
cd codeAIPhishing
Cài đặt các phụ thuộc:

bash
Sao chép
pip install -r requirements.txt
Di chuyển vào thư mục app và chạy ứng dụng web:

bash
Sao chép
python app/main.py
Ứng dụng web sẽ chạy trên http://localhost:5000.

Các Mô Hình Phishing:
Phishing Email: Mô hình được huấn luyện trong thư mục Phishing_Email_Models sử dụng tệp train_RF_email.py.

Phishing File (PDF): Mô hình huấn luyện được thực hiện trong thư mục Phishing_File_Models với tệp train_random_forest.py.

Phishing URL: Mô hình huấn luyện cho phát hiện phishing qua URL nằm trong thư mục Phishing_URL_Models, với tệp rf_url_model.py.

Phishing Image: Mô hình sử dụng CNN trong thư mục Phishing_Image_Models.

Giao Diện Chrome Extension:
Trong thư mục static/, tải tệp app.js và styles.css vào Chrome Extension để kiểm tra phishing URL trực tiếp khi duyệt web.

Đánh Giá Mô Hình
Các mô hình được đánh giá bằng các chỉ số sau:

Accuracy: Tỷ lệ phân loại đúng trên tổng số mẫu.

Precision: Tỷ lệ phân loại đúng các mẫu "Phishing".

Recall: Tỷ lệ các mẫu "Phishing" dự đoán đúng trên tổng số mẫu thực tế là "Phishing".

Kết quả ví dụ của mô hình Random Forest trên các loại dữ liệu phishing:

Mô Hình	Accuracy	Precision	Recall
Phishing URL	93.0%	95.0%	95.0%
Phishing PDF	99.0%	99.0%	99.0%
Phishing Email	99.0%	100%	99.0%
Phishing Hình ảnh	88.0%	85.0%	87.0%

Kết Luận
Hệ thống này chứng minh sức mạnh của các kỹ thuật học máy, đặc biệt là Random Forest và CNN, trong việc phát hiện các mối đe dọa phishing qua nhiều dạng dữ liệu khác nhau như URL, email, PDF và hình ảnh. Dự án hướng đến việc cải thiện an ninh mạng bằng cách cung cấp một giải pháp hiệu quả và dễ dàng triển khai để phát hiện phishing.

Cộng Tác
Hãy thoải mái tạo nhánh mới từ kho lưu trữ này, đóng góp các cải tiến hoặc đề xuất các tính năng mới. Vui lòng đảm bảo tuân theo hướng dẫn mã nguồn của dự án và cung cấp đầy đủ tài liệu cho các thay đổi.

Giấy Phép
Dự án này được cấp phép dưới Giấy phép MIT - xem chi tiết trong tệp LICENSE.
