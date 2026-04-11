"""
Giao diện Streamlit — pipeline demo với các hàm DUMMY (rỗng / giả lập).
Sau này thay thế bằng import thật từ src/preprocessing.py, segmentation.py, ...
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import cv2
import numpy as np
import streamlit as st

# --- Cấu hình pipeline (theo bảng I/O dự án) ---
SEG_W, SEG_H = 800, 400
COLOR_VEC_LEN = 90
SHAPE_VEC_LEN = 100

MENH_GIA_CLASSES = [
    "1.000 VND",
    "2.000 VND",
    "5.000 VND",
    "10.000 VND",
    "20.000 VND",
    "50.000 VND",
    "100.000 VND",
    "200.000 VND",
    "500.000 VND",
]


def dummy_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """
    Hàm rỗng: giả lập tiền xử lý (CLAHE, blur, ...).
    Hiện tại chỉ trả về bản sao ảnh đầu vào (BGR).
    """
    return image_bgr.copy()


def dummy_segment(image_bgr: np.ndarray) -> np.ndarray:
    """
    Hàm rỗng: giả lập phân đoạn + perspective → ảnh chuẩn 800×400 (W×H).
    Thực tế: resize ảnh gốc về kích thước cố định.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((SEG_H, SEG_W, 3), dtype=np.uint8)
    return cv2.resize(image_bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)


def dummy_color_features(cropped_bgr: np.ndarray) -> np.ndarray:
    """
    Hàm rỗng: vector đặc trưng màu (HSV histogram 3 vùng) — độ dài 90.
    Hiện tại: vector ngẫu nhiên có thể tái lập theo nội dung ảnh (seed từ hash).
    """
    seed = int(hashlib.md5(cropped_bgr.tobytes()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.default_rng(seed)
    return rng.random(COLOR_VEC_LEN, dtype=np.float32)


def dummy_shape_features(cropped_bgr: np.ndarray) -> np.ndarray:
    """
    Hàm rỗng: vector SIFT + BoW — độ dài 100.
    Hiện tại: vector ngẫu nhiên có seed từ hash (khác color để không trùng).
    """
    h = hashlib.sha256(cropped_bgr.tobytes()).digest()
    seed = int.from_bytes(h[:4], "big") % (2**31)
    rng = np.random.default_rng(seed + 1)
    return rng.random(SHAPE_VEC_LEN, dtype=np.float32)


def dummy_predict(
    color_vec: np.ndarray, shape_vec: np.ndarray
) -> Tuple[str, float]:
    """
    Hàm rỗng: dự đoán mệnh giá từ hai vector (chưa load .pkl thật).
    Trả về (nhãn, độ tin cậy giả).
    """
    blob = np.concatenate([color_vec, shape_vec]).astype(np.float64).tobytes()
    idx = int(hashlib.sha256(blob).hexdigest(), 16) % len(MENH_GIA_CLASSES)
    conf_seed = int(hashlib.md5(blob).hexdigest()[:8], 16)
    confidence = 0.55 + (conf_seed % 4000) / 10000.0
    return MENH_GIA_CLASSES[idx], float(confidence)


def run_pipeline_dummy(image_bgr: np.ndarray) -> dict:
    """Chạy tuần tự toàn bộ bước dummy, trả dict để hiển thị UI."""
    pre = dummy_preprocess(image_bgr)
    cropped = dummy_segment(pre)
    v_color = dummy_color_features(cropped)
    v_shape = dummy_shape_features(cropped)
    label, conf = dummy_predict(v_color, v_shape)
    return {
        "preprocessed_bgr": pre,
        "cropped_bgr": cropped,
        "color_vector": v_color,
        "shape_vector": v_shape,
        "label": label,
        "confidence": conf,
    }


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được ảnh. Dùng JPG/PNG hợp lệ.")
    return img


def main() -> None:
    st.set_page_config(
        page_title="Nhận diện mệnh giá (demo dummy)",
        layout="wide",
    )
    st.title("Nhận diện mệnh giá tiền Việt Nam — Demo (hàm rỗng)")
    st.caption(
        "Pipeline: Tiền xử lý → Phân đoạn (800×400) → Đặc trưng màu (90) "
        "→ Đặc trưng hình dạng (100) → Dự đoán. Các bước hiện là placeholder."
    )

    f = st.file_uploader("Tải ảnh tiền (có nền cũng được)", type=["jpg", "jpeg", "png", "webp"])
    if f is None:
        st.info("Chọn một file ảnh để chạy pipeline dummy.")
        return

    try:
        image_bgr = decode_uploaded_image(f.getvalue())
    except ValueError as e:
        st.error(str(e))
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Chạy pipeline (dummy)", type="primary"):
        with st.spinner("Đang xử lý..."):
            out = run_pipeline_dummy(image_bgr)

        with col2:
            st.subheader("Ảnh sau phân đoạn (resize dummy 800×400)")
            st.image(
                cv2.cvtColor(out["cropped_bgr"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        st.divider()
        c3, c4, c5 = st.columns(3)
        with c3:
            st.metric("Vector màu (dummy)", f"{len(out['color_vector'])} phần tử")
            st.write("5 phần tử đầu:", out["color_vector"][:5])
        with c4:
            st.metric("Vector hình dạng (dummy)", f"{len(out['shape_vector'])} phần tử")
            st.write("5 phần tử đầu:", out["shape_vector"][:5])
        with c5:
            st.metric("Dự đoán (dummy)", out["label"])
            st.metric("Độ tin cậy (giả)", f"{out['confidence']:.2%}")

        with st.expander("Chi tiết vector (numpy)"):
            st.write("color_vector.shape", out["color_vector"].shape)
            st.write("shape_vector.shape", out["shape_vector"].shape)


if __name__ == "__main__":
    main()
