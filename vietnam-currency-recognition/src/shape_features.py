import cv2
import numpy as np
import os

def _to_gray(image_input):
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        return image
    if isinstance(image_input, np.ndarray):
        return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim == 3 else image_input
    return None


def extract_sift_features(image_input):
    gray_img = _to_gray(image_input)
    if gray_img is None: return None, None
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_img, None)

def count_good_matches(kp_test, des_test, kp_template, des_template, ratio_thresh=0.75):
    """
    Hàm đối sánh nâng cao: Lowe's Ratio Test + RANSAC lọc nhiễu hình học.
    """
    if des_test is None or des_template is None or len(des_test) < 4 or len(des_template) < 4:
        return 0
        
    # 1. So khớp bằng FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_test, des_template, k=2)
    
    # 2. Bước lọc 1: Lowe's Ratio Test
    good_matches_list = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches_list.append(m)
            
    # 3. Bước lọc 2: RANSAC 
    # Cần ít nhất 4 điểm để tìm ma trận Homography
    if len(good_matches_list) >= 4:
        src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
        
        # Tìm ma trận Homography và mặt nạ các điểm inliers (điểm đúng quy luật)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        
        if mask is not None:
            # Trả về số lượng điểm inliers thực sự
            return int(np.sum(mask))
            
    return len(good_matches_list)


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