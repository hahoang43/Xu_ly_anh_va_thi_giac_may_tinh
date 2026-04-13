from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

DEFAULT_VECTOR_LENGTH = 100
DEFAULT_MAX_KEYPOINTS = 500
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_DESCRIPTORS_TOTAL = 50000


def _to_gray(image_input: str | np.ndarray) -> np.ndarray:
    if isinstance(image_input, (str, os.PathLike)):
        image = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Khong the doc anh: {image_input}")
        return image
    if isinstance(image_input, np.ndarray):
        image = image_input.copy()
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Anh dau vao khong hop le")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray
    raise TypeError("image_input phai la duong dan anh hoac numpy.ndarray")


def extract_sift_descriptors(
    image_input: str | np.ndarray,
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
    contrast_threshold: float = 0.01,
    edge_threshold: float = 10,
    sigma: float = 1.2,
) -> np.ndarray:
    gray = _to_gray(image_input)
    sift = cv2.SIFT_create(
        nfeatures=max_keypoints,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or descriptors.size == 0:
        return np.zeros((0, 128), dtype=np.float32)
    return descriptors.astype(np.float32)


def build_bow_vocabulary(
    images: Sequence[str | np.ndarray],
    n_clusters: int = DEFAULT_VECTOR_LENGTH,
    max_keypoints_per_image: int = DEFAULT_MAX_KEYPOINTS,
    max_descriptors_total: int = DEFAULT_MAX_DESCRIPTORS_TOTAL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    descriptor_blocks = []
    for image in images:
        descriptors = extract_sift_descriptors(image, max_keypoints=max_keypoints_per_image)
        if descriptors.shape[0] > 0:
            descriptor_blocks.append(descriptors)

    if not descriptor_blocks:
        return np.zeros((n_clusters, 128), dtype=np.float32)

    all_descriptors = np.vstack(descriptor_blocks).astype(np.float32)

    if all_descriptors.shape[0] > max_descriptors_total:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(all_descriptors.shape[0], size=max_descriptors_total, replace=False)
        all_descriptors = all_descriptors[idx]

    effective_clusters = min(n_clusters, all_descriptors.shape[0])
    if effective_clusters == 0:
        return np.zeros((n_clusters, 128), dtype=np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        batch_size=min(4096, max(256, effective_clusters * 10)),
        n_init=5,
    )
    kmeans.fit(all_descriptors)

    vocabulary = kmeans.cluster_centers_.astype(np.float32)
    if effective_clusters < n_clusters:
        pad = np.zeros((n_clusters - effective_clusters, 128), dtype=np.float32)
        vocabulary = np.vstack([vocabulary, pad])
    return vocabulary


def save_vocabulary(vocabulary: np.ndarray, vocab_path: str | os.PathLike) -> None:
    vocab = np.asarray(vocabulary, dtype=np.float32)
    if vocab.ndim != 2 or vocab.shape[1] != 128:
        raise ValueError("Vocabulary phai co shape (K, 128)")
    path = Path(vocab_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vocab)


def load_vocabulary(vocab_path: str | os.PathLike, n_clusters: int | None = None) -> np.ndarray:
    vocabulary = np.load(vocab_path).astype(np.float32)
    if vocabulary.ndim != 2 or vocabulary.shape[1] != 128:
        raise ValueError("Vocabulary phai co shape (K, 128)")
    if n_clusters is None:
        return vocabulary
    if vocabulary.shape[0] > n_clusters:
        return vocabulary[:n_clusters]
    if vocabulary.shape[0] < n_clusters:
        pad = np.zeros((n_clusters - vocabulary.shape[0], 128), dtype=np.float32)
        return np.vstack([vocabulary, pad])
    return vocabulary


def _normalize_vocabulary(vocabulary: np.ndarray | None, n_clusters: int) -> np.ndarray:
    if vocabulary is None:
        return np.zeros((n_clusters, 128), dtype=np.float32)
    vocab = np.asarray(vocabulary, dtype=np.float32)
    if vocab.ndim != 2 or vocab.shape[1] != 128:
        raise ValueError("Vocabulary phai co shape (K, 128)")
    if vocab.shape[0] > n_clusters:
        return vocab[:n_clusters]
    if vocab.shape[0] < n_clusters:
        pad = np.zeros((n_clusters - vocab.shape[0], 128), dtype=np.float32)
        return np.vstack([vocab, pad])
    return vocab


def _compute_bow_histogram(descriptors: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    n_clusters = vocabulary.shape[0]
    if n_clusters == 0 or descriptors.shape[0] == 0:
        return np.zeros((n_clusters,), dtype=np.float32)
    if not np.any(vocabulary):
        return np.zeros((n_clusters,), dtype=np.float32)

    dists = np.linalg.norm(descriptors[:, None, :] - vocabulary[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    hist = np.bincount(nearest, minlength=n_clusters).astype(np.float32)

    total = hist.sum()
    if total > 0:
        hist /= total

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist.astype(np.float32)


def get_shape_bow_vector(
    image_input: str | np.ndarray,
    vocabulary: np.ndarray | None = None,
    vocab_path: str | os.PathLike | None = None,
    n_clusters: int = DEFAULT_VECTOR_LENGTH,
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
    contrast_threshold: float = 0.01,
    edge_threshold: float = 10,
    sigma: float = 1.2,
) -> np.ndarray:
    if vocabulary is None and vocab_path is not None:
        vocabulary = load_vocabulary(vocab_path, n_clusters=n_clusters)
    vocabulary = _normalize_vocabulary(vocabulary, n_clusters)
    descriptors = extract_sift_descriptors(
        image_input,
        max_keypoints=max_keypoints,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )
    return _compute_bow_histogram(descriptors, vocabulary)


def fit_and_save_bow_vocabulary(
    images: Sequence[str | np.ndarray],
    vocab_path: str | os.PathLike,
    n_clusters: int = DEFAULT_VECTOR_LENGTH,
    max_keypoints_per_image: int = DEFAULT_MAX_KEYPOINTS,
    max_descriptors_total: int = DEFAULT_MAX_DESCRIPTORS_TOTAL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    vocabulary = build_bow_vocabulary(
        images=images,
        n_clusters=n_clusters,
        max_keypoints_per_image=max_keypoints_per_image,
        max_descriptors_total=max_descriptors_total,
        random_state=random_state,
    )
    save_vocabulary(vocabulary, vocab_path)
    return vocabulary