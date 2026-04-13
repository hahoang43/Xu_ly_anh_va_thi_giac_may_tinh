"""
Train model nhận diện mệnh giá.

Ưu tiên:
1) Dữ liệu ảnh thật theo cấu trúc data/raw/<label>/*.jpg|png
   -> trích đặc trưng 190 chiều (90 màu + 100 shape BoW)
2) Nếu không có ảnh thật, fallback dùng mock_features.csv (để script vẫn chạy được)
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from color_features import get_hsv_histogram
from shape_features import (
    DEFAULT_VECTOR_LENGTH,
    build_bow_vocabulary,
    get_shape_bow_vector,
    save_vocabulary,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

LABEL_COL = "label"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
COLOR_LEN = 90
SHAPE_LEN = DEFAULT_VECTOR_LENGTH
TOTAL_LEN = COLOR_LEN + SHAPE_LEN


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sort_feature_like_columns(names: Sequence[str]) -> List[str]:
    def key(name: str) -> Tuple[int, str]:
        m = re.match(r"^feature_(\d+)$", name.strip(), re.IGNORECASE)
        if m:
            return (0, int(m.group(1)))
        return (1, name)

    return sorted(list(names), key=key)


def _select_feature_column_names(headers: Sequence[str]) -> List[str]:
    raw = [h.strip() for h in headers if h and h.strip().lower() != LABEL_COL]
    lower_map = {h: h.strip().lower() for h in raw}
    cv3 = [h for h in raw if lower_map[h].startswith("cv3")]
    cv4 = [h for h in raw if lower_map[h].startswith("cv4")]
    if cv3 or cv4:
        return sorted(cv3, key=lambda x: lower_map[x]) + sorted(cv4, key=lambda x: lower_map[x])
    return _sort_feature_like_columns(raw)


def load_mock_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    errors: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")
        if LABEL_COL not in reader.fieldnames:
            raise ValueError(f"Thiếu cột nhãn '{LABEL_COL}'.")
        feature_cols = _select_feature_column_names(reader.fieldnames)
        rows_x: List[List[float]] = []
        rows_y: List[str] = []
        for i, row in enumerate(reader, start=2):
            try:
                rows_x.append([float(row[c]) for c in feature_cols])
                rows_y.append(str(row[LABEL_COL]).strip())
            except (KeyError, TypeError, ValueError) as e:
                errors.append(f"Dòng {i}: bỏ qua ({e})")
    if not rows_x:
        raise ValueError("Không có dòng hợp lệ trong CSV.")
    return np.asarray(rows_x, dtype=np.float64), np.asarray(rows_y), feature_cols, errors


def _simple_segment(image_bgr: np.ndarray, width: int = 800, height: int = 400) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def _list_image_samples(data_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    if not data_dir.is_dir():
        return samples
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name.strip()
        for img in class_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                samples.append((img, label))
    return samples


def _extract_color_shape_vector(image_path: Path, vocabulary: np.ndarray) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    crop = _simple_segment(image_bgr)
    v_color = get_hsv_histogram(crop, bins=10).astype(np.float32)
    v_shape = get_shape_bow_vector(crop, vocabulary=vocabulary, n_clusters=SHAPE_LEN).astype(np.float32)
    return np.concatenate([v_color, v_shape], axis=0)


def load_image_dataset(
    data_dir: Path,
    vocab_path: Path,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    samples = _list_image_samples(data_dir)
    if not samples:
        raise ValueError(f"Không tìm thấy ảnh trong {data_dir}")

    image_paths = [p for p, _ in samples]
    labels = np.asarray([lb for _, lb in samples])
    train_idx, _ = train_test_split(
        np.arange(len(samples)),
        test_size=0.25,
        random_state=random_state,
        stratify=labels,
    )
    train_images_for_vocab = [image_paths[i] for i in train_idx]
    vocab = build_bow_vocabulary(train_images_for_vocab, n_clusters=SHAPE_LEN, random_state=random_state)
    save_vocabulary(vocab, vocab_path)

    features: List[np.ndarray] = []
    y: List[str] = []
    skipped = 0
    for p, lb in samples:
        try:
            vec = _extract_color_shape_vector(p, vocabulary=vocab)
            if vec.shape[0] != TOTAL_LEN:
                raise ValueError(f"Vector length {vec.shape[0]} != {TOTAL_LEN}")
            features.append(vec.astype(np.float64))
            y.append(lb)
        except Exception:
            skipped += 1
    if not features:
        raise ValueError("Không trích được feature nào từ ảnh thật.")
    X = np.vstack(features)
    feature_cols = [f"feature_{i+1}" for i in range(TOTAL_LEN)]
    meta = {
        "data_dir": str(data_dir.resolve()),
        "vocab_path": str(vocab_path.resolve()),
        "n_raw_images": len(samples),
        "n_skipped_images": skipped,
    }
    return X, np.asarray(y), feature_cols, meta


def build_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=random_state)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")),
            ]
        ),
    }


def train_and_pick_best(
    X: np.ndarray,
    y_encoded: np.ndarray,
    random_state: int,
    test_size: float,
) -> Tuple[str, Pipeline, Dict[str, object], Dict[str, Dict[str, float]]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    metrics_all: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_tuple = (-1.0, -1.0)
    best_model: Pipeline | None = None
    for name, pipe in build_models(random_state).items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        metrics_all[name] = {"accuracy": float(acc), "f1_macro": float(f1)}
        if (acc, f1) > best_tuple:
            best_tuple = (acc, f1)
            best_name = name
            best_model = pipe
    assert best_model is not None
    return (
        best_name,
        best_model,
        {"best_model": best_name, "accuracy": metrics_all[best_name]["accuracy"], "f1_macro": metrics_all[best_name]["f1_macro"]},
        metrics_all,
    )


def save_artifact(
    path: Path,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_columns: Sequence[str],
    meta: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_columns": list(feature_columns),
        "meta": meta,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train SVM vs KNN for currency recognition")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw", help="Thư mục ảnh: data/raw/<label>/*.jpg")
    parser.add_argument("--csv", type=Path, default=root / "data" / "mock" / "mock_features.csv")
    parser.add_argument("--out", type=Path, default=root / "models" / "best_classifier.pkl")
    parser.add_argument("--vocab-out", type=Path, default=root / "models" / "shape_vocab.npy")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    source_mode = "mock_csv"
    source_meta: Dict[str, object] = {}
    if args.data_dir.is_dir() and any(p.is_dir() for p in args.data_dir.iterdir()):
        try:
            X, y_str, feature_cols, source_meta = load_image_dataset(
                data_dir=args.data_dir,
                vocab_path=args.vocab_out,
                random_state=args.random_state,
            )
            source_mode = "real_images"
        except Exception as e:
            print(f"[CẢNH BÁO] Train từ ảnh thật thất bại: {e}")
            print("[INFO] Fallback sang mock_features.csv")
            X, y_str, feature_cols, warnings = load_mock_dataset(args.csv)
            for msg in warnings[:5]:
                print(msg)
    else:
        X, y_str, feature_cols, warnings = load_mock_dataset(args.csv)
        for msg in warnings[:5]:
            print(msg)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    best_name, best_pipe, summary, metrics_all = train_and_pick_best(X, y, args.random_state, args.test_size)
    # Refit trên full data để deployment
    best_pipe.fit(X, y)

    meta = {
        "source_mode": source_mode,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": feature_cols,
        "classes": [str(c) for c in le.classes_],
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics_all,
        "best": summary,
    }
    meta.update(source_meta)
    save_artifact(args.out, best_pipe, le, feature_cols, meta)

    print("--- Train done ---")
    print(f"Source mode: {source_mode}")
    for mname, m in metrics_all.items():
        print(f"  {mname}: accuracy={m['accuracy']:.4f}, f1_macro={m['f1_macro']:.4f}")
    print(f"Best model: {best_name}")
    print(f"Saved model: {args.out.resolve()}")
    if source_mode == "real_images":
        print(f"Saved vocab: {args.vocab_out.resolve()}")


if __name__ == "__main__":
    main()
