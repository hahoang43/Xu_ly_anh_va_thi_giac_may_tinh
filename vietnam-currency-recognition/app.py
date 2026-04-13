"""
Giao diện Streamlit — pipeline demo với các hàm DUMMY (rỗng / giả lập).
Sau này thay thế bằng import thật từ src/preprocessing.py, segmentation.py, ...
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
from src.color_features import get_hsv_histogram

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
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_classifier.pkl"
EXPECTED_FEATURE_LEN = COLOR_VEC_LEN + SHAPE_VEC_LEN


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


@st.cache_resource
def load_trained_model() -> dict | None:
    """Load artifact từ train_model.py nếu đã tồn tại."""
    if not MODEL_PATH.is_file():
        return None
    try:
        with MODEL_PATH.open("rb") as f:
            payload = pickle.load(f)
        return payload
    except Exception:
        return None


def is_model_compatible(payload: dict | None) -> bool:
    if payload is None:
        return False
    feature_columns = payload.get("feature_columns", [])
    return len(feature_columns) == EXPECTED_FEATURE_LEN


def build_feature_for_model(cropped_bgr: np.ndarray, expected_len: int) -> np.ndarray:
    """
    Tạo vector đưa vào model:
    - Ưu tiên đặc trưng màu thật từ src.color_features (90).
    - Ghép thêm shape dummy để đủ độ dài nếu cần.
    - Nếu dư thì cắt bớt.
    """
    color_vec = get_hsv_histogram(cropped_bgr, bins=10).astype(np.float32)
    shape_vec = dummy_shape_features(cropped_bgr).astype(np.float32)
    merged = np.concatenate([color_vec, shape_vec], axis=0)
    if merged.shape[0] == expected_len:
        return merged
    if merged.shape[0] > expected_len:
        return merged[:expected_len]
    pad = np.zeros((expected_len - merged.shape[0],), dtype=np.float32)
    return np.concatenate([merged, pad], axis=0)


def predict_with_trained_model(cropped_bgr: np.ndarray, payload: dict) -> Tuple[str, float, str]:
    """Predict bằng model .pkl. Trả label, confidence, ghi chú căn chỉnh vector."""
    pipe = payload.get("pipeline")
    le = payload.get("label_encoder")
    feature_columns = payload.get("feature_columns", [])
    expected_len = len(feature_columns)
    if expected_len <= 0:
        raise ValueError("Model artifact thiếu feature_columns.")
    if expected_len != EXPECTED_FEATURE_LEN:
        raise ValueError(
            f"Model có {expected_len} features, không khớp chuẩn {EXPECTED_FEATURE_LEN}."
        )

    x = build_feature_for_model(cropped_bgr, expected_len).reshape(1, -1)
    pred_encoded = pipe.predict(x)[0]
    label = str(le.inverse_transform([pred_encoded])[0])
    note = f"Model cần {expected_len} features"

    confidence = 0.0
    if hasattr(pipe, "predict_proba"):
        try:
            prob = pipe.predict_proba(x)[0]
            confidence = float(np.max(prob))
        except Exception:
            confidence = 0.0
    elif hasattr(pipe, "decision_function"):
        try:
            dec = np.atleast_1d(pipe.decision_function(x)).astype(float)
            if dec.ndim > 1:
                dec = dec[0]
            exp = np.exp(dec - np.max(dec))
            prob = exp / np.sum(exp)
            confidence = float(np.max(prob))
        except Exception:
            confidence = 0.0

    if confidence <= 0:
        confidence = 0.5

    if "." not in label:
        try:
            label = f"{int(label):,}".replace(",", ".") + " VND"
        except ValueError:
            pass
    return label, confidence, note


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
    model_note = "Đang dùng dummy"
    payload = load_trained_model()
    if is_model_compatible(payload):
        try:
            label, conf, model_note = predict_with_trained_model(cropped, payload)
            v_color = get_hsv_histogram(cropped, bins=10).astype(np.float32)
        except Exception:
            label, conf = dummy_predict(v_color, v_shape)
            model_note = "Model lỗi, fallback dummy"
    else:
        label, conf = dummy_predict(v_color, v_shape)
        if payload is not None:
            model_note = "Model không tương thích, fallback dummy"
    return {
        "preprocessed_bgr": pre,
        "cropped_bgr": cropped,
        "color_vector": v_color,
        "shape_vector": v_shape,
        "label": label,
        "confidence": conf,
        "model_note": model_note,
    }


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được ảnh. Dùng JPG/PNG hợp lệ.")
    return img


def main() -> None:
    st.set_page_config(
        page_title="Nhận diện mệnh giá tiền Việt Nam",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1680px;
                padding-top: 1.8rem;
                padding-bottom: 1.6rem;
            }
            .main-title {
                text-align: center;
                font-size: 46px;
                font-weight: 700;
                letter-spacing: 0.4px;
                margin-bottom: 0.35rem;
                margin-top: 0.45rem;
                line-height: 1.35;
                color: #0f172a;
                overflow: visible;
            }
            .sub-title {
                text-align: center;
                font-size: 21px;
                color: #475569;
                margin-bottom: 1.15rem;
            }
            .section-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 20px;
                min-height: 155px;
            }
            .section-title {
                margin: 0 0 8px;
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            .hint {
                color: #64748b;
                font-size: 16px;
                margin-bottom: 12px;
            }
            div.stButton > button {
                height: 58px;
                border-radius: 10px;
                font-weight: 700;
                letter-spacing: 0.2px;
                font-size: 21px;
            }
            [data-testid="stFileUploaderDropzone"] {
                padding-top: 30px;
                padding-bottom: 30px;
                border-radius: 14px;
                min-height: 110px;
            }
            [data-testid="stFileUploaderDropzone"] small {
                font-size: 16px !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                height: 44px !important;
                font-size: 18px !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 42px;
            }
            .result-card {
                border: 1px solid #dbe3ef;
                border-radius: 14px;
                padding: 22px 24px;
                background: #ffffff;
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
                min-height: 210px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .result-title {
                margin: 0 0 10px;
                font-size: 20px;
                color: #64748b;
                font-weight: 600;
                min-height: 24px;
            }
            .result-value {
                margin: 0;
                font-size: 50px;
                color: #0f172a;
                font-weight: 700;
                line-height: 1.15;
                white-space: nowrap;
            }
            .result-panel {
                min-height: 0;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="main-title">NHẬN DIỆN MỆNH GIÁ TIỀN VIỆT NAM</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Input: Ảnh tiền Việt Nam | Output: Mệnh giá dự đoán + Mức độ tin cậy</div>',
        unsafe_allow_html=True,
    )
    model_payload = load_trained_model()
    if model_payload is None:
        st.caption("Chưa có model chuẩn, hệ thống đang chạy chế độ dummy.")

    f = st.file_uploader("Chọn ảnh đầu vào", type=["jpg", "jpeg", "png", "webp"])

    if f is None:
        st.info("Chọn 1 ảnh để hệ thống dự đoán mệnh giá.")
        return

    try:
        image_bgr = decode_uploaded_image(f.getvalue())
    except ValueError as e:
        st.error(str(e))
        return

    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with right:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        if st.button("Nhận diện mệnh giá", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý..."):
                st.session_state["predict_out"] = run_pipeline_dummy(image_bgr)

        out = st.session_state.get("predict_out")
        if out is not None:
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="result-title">Mệnh giá dự đoán</p>
                        <p class="result-value">{out["label"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="result-title">Mức độ tin cậy</p>
                        <p class="result-value">{out['confidence']:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
