import cv2
import numpy as np
import os

def get_hsv_histogram(image_input, bins=10):
    """
    Hàm nhận vào ảnh tiền đã cắt, chia làm 3 phần và tính HSV Histogram.
    
    Args:
        image_input: Có thể là đường dẫn ảnh (string) hoặc mảng pixel numpy (Mock Data).
        bins (int): Số lượng bin cho mỗi kênh màu (mặc định 10 để tổng vector = 90).
        
    Returns:
        numpy.ndarray: Vector đặc trưng 1 chiều kiểu số thực, độ dài 90.
    """
    
    # 1. Xử lý Input (Hỗ trợ cả đường dẫn file hoặc mảng numpy)
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("Input phải là đường dẫn ảnh hoặc mảng numpy.")

    # 2. Chuyển đổi không gian màu từ BGR sang HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv_img.shape

    # 3. Chia ảnh làm 3 vùng: Trái, Giữa, Phải theo chiều rộng
    w_third = width // 3
    
    left_region = hsv_img[:, :w_third]
    middle_region = hsv_img[:, w_third:2*w_third]
    right_region = hsv_img[:, 2*w_third:]
    
    regions = [left_region, middle_region, right_region]
    feature_vector = []

    # 4. Tính Histogram cho từng vùng
    for region in regions:
        # Tính histogram cho từng kênh H, S, V riêng biệt
        hist_h = cv2.calcHist([region], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([region], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([region], [2], None, [bins], [0, 256])

        # Chuẩn hóa (Normalize) histogram
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Nối (flatten) các kênh H, S, V của vùng hiện tại
        region_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        # Thêm vào mảng vector tổng
        feature_vector.extend(region_features)

    # 5. Trả về mảng số thực (float32)
    return np.array(feature_vector, dtype=np.float32)

# ==========================================
# KHU VỰC TEST
# ==========================================
if __name__ == "__main__":
    print("--- Đang kiểm tra hàm tính HSV Histogram ---")
    
    # Tự động lấy đường dẫn của thư mục hiện tại (thư mục 'src')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Trỏ ngược ra ngoài 1 cấp, vào 'data/raw/1000' để lấy ảnh test thật
    test_image_path = os.path.join(current_dir, "..", "data", "raw", "1000", "1000_sau_gap.jpg")
    
    # Kiểm tra xem có tìm thấy ảnh thật không
    if os.path.exists(test_image_path):
        print(f"[OK] Tìm thấy ảnh thật để test: {test_image_path}")
        try:
            vector_dac_trung = get_hsv_histogram(test_image_path, bins=10)
            print(f"Kiểu dữ liệu trả về: {type(vector_dac_trung)}")
            print(f"Độ dài vector: {len(vector_dac_trung)} phần tử")
            print("Giá trị vector (5 phần tử đầu):", vector_dac_trung[:5])
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh thật: {e}")
            
    else:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh thật tại: {test_image_path}")
        print("-> Chuyển sang dùng Mock Data giả lập...")
        
        mock_image = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)
        vector_dac_trung = get_hsv_histogram(mock_image, bins=10)
        
        print(f"Độ dài vector: {len(vector_dac_trung)} phần tử")