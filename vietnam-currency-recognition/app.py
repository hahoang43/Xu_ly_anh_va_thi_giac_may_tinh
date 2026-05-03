import cv2
import numpy as np
import streamlit as st
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import các hàm từ src/main.py
from main import nap_mau_tu_dong, nhan_dien_tien
from segmentation import phan_doan_va_nan_chinh
from preprocessing import tien_xu_ly_anh
# ====================================================================
# 3. GIAO DIỆN STREAMLIT
# ====================================================================
def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được ảnh. Vui lòng dùng JPG/PNG hợp lệ.")
    return img
@st.cache_resource
def load_templates_once():
    return nap_mau_tu_dong('data/raw')
def main() -> None:
    st.set_page_config(page_title="Nhận diện tiền VNĐ", layout="wide")
    st.markdown(
        """
        <style>
            .main-title {text-align: center; font-size: 40px; font-weight: 700; color: #0f172a;}
            .sub-title {text-align: center; font-size: 18px; color: #475569; margin-bottom: 20px;}
            .result-card {border: 1px solid #dbe3ef; border-radius: 14px; padding: 20px; background: #ffffff; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center;}
            .result-title {font-size: 18px; color: #64748b; font-weight: 600;}
            .result-value {font-size: 40px; color: #0f172a; font-weight: 700;}
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-title">HỆ THỐNG NHẬN DIỆN MỆNH GIÁ TIỀN VIỆT NAM</h1>', unsafe_allow_html=True)

    if "templates" not in st.session_state:
        with st.spinner("Hệ thống đang nạp thư viện"):
            st.session_state["templates"] = load_templates_once()
    
    templates = st.session_state["templates"]
        
    if len(templates) == 0:
        st.error(" Không tìm thấy thư mục 'data/raw' hoặc ảnh mẫu. Vui lòng kiểm tra cấu trúc thư mục!")
        return

# Thay thế đoạn lỗi bằng 2 dòng này:
    f = st.file_uploader("📸 Chọn ảnh tờ tiền cần nhận diện", type=["jpg", "jpeg", "png"])
    is_already_cropped = False # Mặc định luôn là False để hệ thống tự dò biên

    if f is not None:
        image_bgr = decode_uploaded_image(f.getvalue())

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.markdown("**1. Ảnh gốc tải lên**")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.markdown("**2. Phân đoạn & Nắn phẳng**")
            if "img_cat" in st.session_state:
                st.image(cv2.cvtColor(st.session_state["img_cat"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with col3:
            st.markdown("**3. Tiền xử lý (CLAHE)**")
            if "img_sach" in st.session_state:
                st.image(cv2.cvtColor(st.session_state["img_sach"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with col4:
            st.markdown("**4. Kết quả nhận diện**")
            if "result" in st.session_state:
                label, conf, exec_time = st.session_state["result"]
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="result-title">Mệnh giá dự đoán</p>
                        <p class="result-value" style="color: #2e7d32;">{label}</p>
                        <hr style="margin: 10px 0;">
                    </div>
                    """, unsafe_allow_html=True
                )

        if st.button("Bắt đầu Nhận diện", type="primary", use_container_width=True):
            with st.spinner("Đang chạy thuật toán Pipeline..."):
                start_time = time.time()
                
                # 1. Tạo ảnh hiển thị cho Bước 2 và Bước 3 (Vì file main không trả về ảnh)
                if is_already_cropped:
                    img_cat = cv2.resize(image_bgr, (800, 400))
                else:
                    img_cat = phan_doan_va_nan_chinh(image_bgr)
                    if img_cat is None: img_cat = cv2.resize(image_bgr, (800, 400))
                
                img_sach = tien_xu_ly_anh(img_cat)
                
                # 2. Lưu ảnh tạm để hàm nhan_dien_tien trong main.py đọc được (vì main nhận path)
                temp_path = "temp_for_main.jpg"
                cv2.imwrite(temp_path, image_bgr)
                
                # 3. GỌI ĐÚNG HÀM TỪ MAIN.PY (Thay thế cho predict_image)
                # Hàm này nhận (path, templates) và trả về (label, info)
                label, info = nhan_dien_tien(temp_path, templates)
                
                # 4. Lưu kết quả vào session_state để hiển thị lên giao diện
                st.session_state["img_cat"] = img_cat
                st.session_state["img_sach"] = img_sach
                
                # Vì main không trả về confidence, mình để mặc định hoặc lấy từ info nếu có
                conf = 0.99 if label != "Không xác định" else 0.0
                st.session_state["result"] = (label, conf, time.time() - start_time)
                
                # Dọn dẹp file tạm và cập nhật giao diện
                if os.path.exists(temp_path): os.remove(temp_path)
                st.rerun()

if __name__ == "__main__":
    main()