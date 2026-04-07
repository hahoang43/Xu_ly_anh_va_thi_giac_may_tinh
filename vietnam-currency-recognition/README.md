# Hệ Thống Nhận Diện Mệnh Giá Tiền Việt Nam 

Đây là mã nguồn của Đồ án môn học **Xử lý ảnh và Thị giác máy tính (121036)** - Trường Đại học Giao thông Vận tải TP. [cite_start]Hồ Chí Minh[cite: 1, 3]. 

Hệ thống ứng dụng các kỹ thuật Thị giác máy tính truyền thống (OpenCV) và Học máy (scikit-learn) để tự động nhận diện mệnh giá tiền Việt Nam,

## 👥 Nhóm Thực Hiện (Nhóm 6 người)
1. [Tên TV1] - MSSV - Quản lý Dữ liệu & Tiền xử lý
2. [Tên TV2] - MSSV - Phân đoạn ảnh & Căn chỉnh
3. [Tên TV3] - MSSV - Trích xuất đặc trưng màu sắc
4. [Tên TV4] - MSSV - Trích xuất đặc trưng bề mặt (ORB)
5. [Tên TV5] - MSSV - Xây dựng mô hình Học máy
6. [Tên TV6] - MSSV - Tích hợp Giao diện & Đánh giá

---

## 🛠 Công Nghệ Sử Dụng
* **Ngôn ngữ:** Python 3.9+
* [cite_start]**Thị giác máy tính:** OpenCV (`cv2`), `scikit-image` [cite: 26]
* **Học máy (Machine Learning):** `scikit-learn` (Sử dụng mô hình phân loại SVM/KNN)
* **Toán học & Ma trận:** `NumPy`
* **Giao diện Demo:** `Streamlit`

---
⚙️ Hướng Dẫn Cài Đặt
    git clone [}
Cài đặt các thư viện yêu cầu
    pip install -r requirements.txt
Hướng Dẫn Chạy Pipeline
Chạy Giao diện Demo trực tiếp:
streamlit run app.py