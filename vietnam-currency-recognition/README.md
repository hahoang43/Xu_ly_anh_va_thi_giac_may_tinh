
# 🇻🇳 Vietnamese Currency Recognition System
**Hệ thống nhận diện mệnh giá tiền Việt Nam **

Hệ thống sử dụng kỹ thuật xử lý ảnh để nhận diện các mệnh giá tiền Polymer Việt Nam. Ứng dụng được xây dựng trên nền tảng **Streamlit**, cho phép người dùng tải ảnh lên và xem kết quả phân tích theo từng bước trong Pipeline xử lý.

## 📋 Tính năng chính
*   **Tiền xử lý nâng cao:** Sử dụng CLAHE để cân bằng độ tương phản và khử nhiễu.
*   **Phân đoạn & Nắn chỉnh:** Tự động dò biên tờ tiền và thực hiện Perspective Transform để đưa về mặt phẳng chuẩn $800 \times 400$.
*   **Hybrid Scoring:** Kết hợp giữa đặc trưng hình thái (**SIFT** - 75%) và màu sắc (**HSV** - 25%) để tối ưu độ chính xác.
*   **Lọc nhiễu RANSAC:** Loại bỏ các điểm khớp sai lệch về mặt hình học, giúp hệ thống đạt độ chính xác cao ngay cả khi ảnh bị nghiêng hoặc che khuất một phần.

## 🏗️ Cấu trúc thư mục
## 🛠️ Hướng dẫn cài đặt

1. **Khởi tạo môi trường:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

3. **Chạy ứng dụng:**
*   **Chạy giao diện Web:**
    ```bash
    streamlit run app.py
    ```
*   **Chạy kiểm thử Offline (Tính Accuracy):**
    ```bash
    python main.py
    ```

## Thuật toán nhận diện
Hệ thống áp dụng công thức tính điểm lai (Hybrid Score) để đưa ra kết luận cuối cùng:

$$Total\_Score = (SIFT\_Score \times 0.75) + (HSV\_Score \times 0.25)$$

Nếu $Total\_Score < 12.0$, hệ thống sẽ trả về kết quả "Không xác định" để đảm bảo tính an toàn cho nhận diện.

## 👥 Thành viên thực hiện

| STT | Họ và Tên | MSSV |
|:---:|:---|:---:|:---|
| 1 | **Võ Hồ Hoàng Hà** | 052304014420 |
| 2 | **Nguyễn Trọng Phú** | 054205000746 |
| 3 | **Phan Hoàng Phúc** | 075305005338 |
| 4 | **Đoàn Tấn Tài** | 052205002254 |
| 5 | **Phạm Minh Lượng** | 2251120367 |
| 6 | **Châu Thế Tùng** | 052205014665 |

> **Lưu ý:** Để hệ thống đạt độ chính xác cao nhất, hãy đảm bảo thư mục `data/raw` chứa ít nhất 4-6 ảnh mẫu cho mỗi mệnh giá (ưu tiên ảnh phẳng, rõ nét).

```