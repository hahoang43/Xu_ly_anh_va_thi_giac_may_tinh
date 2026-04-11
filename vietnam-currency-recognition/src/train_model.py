"""
Huấn luyện phân loại 9 mệnh giá: đọc mock_features.csv (vector CV3 + CV4),
chia Train/Test, so sánh SVM vs KNN, lưu mô hình tốt nhất ra file .pkl.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

LABEL_COL = "label"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sort_feature_like_columns(names: Sequence[str]) -> List[str]:
    """Sắp feature_1, feature_2, ... theo số; còn lại giữ thứ tự alphabet."""

    def key(name: str) -> Tuple[int, str]:
        m = re.match(r"^feature_(\d+)$", name.strip(), re.IGNORECASE)
        if m:
            return (0, int(m.group(1)))
        return (1, name)

    return sorted(list(names), key=key)


def _select_feature_column_names(headers: Sequence[str]) -> List[str]:
    """
    Chọn cột đặc trưng theo ưu tiên:
    1) Nếu có cột bắt đầu bằng cv3 / cv4 (không phân biệt hoa thường) → ghép CV3 rồi CV4.
    2) Ngược lại → mọi cột số (trừ nhãn) — mock hiện tại dùng feature_1..feature_n.
    """
    raw = [h.strip() for h in headers if h and h.strip().lower() != LABEL_COL]
    lower_map = {h: h.strip().lower() for h in raw}

    cv3 = [h for h in raw if lower_map[h].startswith("cv3")]
    cv4 = [h for h in raw if lower_map[h].startswith("cv4")]
    if cv3 or cv4:
        cv3 = sorted(cv3, key=lambda x: lower_map[x])
        cv4 = sorted(cv4, key=lambda x: lower_map[x])
        return cv3 + cv4

    non_label = [h for h in raw if lower_map[h] != LABEL_COL]
    return _sort_feature_like_columns(non_label)


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Đọc CSV → X (float), y (chuỗi nhãn mệnh giá), danh sách cột feature, danh sách lỗi (rỗng nếu ok)."""
    errors: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")

        feature_cols = _select_feature_column_names(reader.fieldnames)
        if LABEL_COL not in reader.fieldnames:
            raise ValueError(f"Thiếu cột nhãn '{LABEL_COL}' trong CSV.")

        rows_x: List[List[float]] = []
        rows_y: List[str] = []
        for i, row in enumerate(reader, start=2):
            try:
                y_val = str(row[LABEL_COL]).strip()
                vec = [float(row[c]) for c in feature_cols]
            except (KeyError, TypeError, ValueError) as e:
                errors.append(f"Dòng {i}: bỏ qua ({e})")
                continue
            rows_x.append(vec)
            rows_y.append(y_val)

    if not rows_x:
        raise ValueError("Không đọc được dòng dữ liệu hợp lệ nào.")

    X = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y)
    return X, y, feature_cols, errors


def build_models(random_state: int) -> Dict[str, Pipeline]:
    """Hai pipeline: chuẩn hóa + bộ phân loại (siêu tham số mặc định, dataset mock nhỏ)."""
    return {
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        metric="euclidean",
                    ),
                ),
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
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
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
        cand = (acc, f1)
        if cand > best_tuple:
            best_tuple = cand
            best_name = name
            best_model = pipe

    assert best_model is not None
    summary = {
        "best_model": best_name,
        "accuracy": metrics_all[best_name]["accuracy"],
        "f1_macro": metrics_all[best_name]["f1_macro"],
    }
    return best_name, best_model, summary, metrics_all


def save_artifact(
    path: Path,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_columns: Sequence[str],
    meta: Dict,
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
    parser = argparse.ArgumentParser(description="Train SVM vs KNN on mock_features.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "data" / "mock" / "mock_features.csv",
        help="Đường dẫn mock_features.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "models" / "best_classifier.pkl",
        help="File .pkl lưu pipeline + LabelEncoder",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    csv_path: Path = args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không thấy file: {csv_path}")

    X, y_str, feature_cols, load_errors = load_dataset(csv_path)
    for msg in load_errors[:10]:
        print(msg)
    if len(load_errors) > 10:
        print(f"... và {len(load_errors) - 10} cảnh báo khác.")

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    if n_classes != 9:
        print(f"Cảnh báo: CSV có {n_classes} lớp (kỳ vọng 9): {list(le.classes_)}")

    best_name, best_pipe, summary, metrics_all = train_and_pick_best(
        X, y, args.random_state, args.test_size
    )

    meta = {
        "csv": str(csv_path.resolve()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": feature_cols,
        "classes": [str(c) for c in le.classes_],
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics_all,
        "best": summary,
    }
    save_artifact(args.out, best_pipe, le, feature_cols, meta)

    print("--- So sánh trên tập test ---")
    for mname, m in metrics_all.items():
        print(f"  {mname}: accuracy={m['accuracy']:.4f}, f1_macro={m['f1_macro']:.4f}")
    print(f"Chọn mô hình tốt nhất: {summary['best_model']}")
    print(f"Đã lưu: {args.out.resolve()}")


if __name__ == "__main__":
    main()
