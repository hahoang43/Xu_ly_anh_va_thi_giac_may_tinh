import cv2
import numpy as np
import os

def get_hsv_histogram(image_input, bins=10):
    """
    Hàm nhận vào ảnh tiền đã cắt, chia làm 3 phần và tính HSV Histogram.
    [ĐÃ NÂNG CẤP]: Tích hợp Bộ lọc chống lóa (Glare Filter) với ngưỡng 245 để không xóa nhầm tờ 2.000đ và 500.000đ.
    """
    # 1. Xử lý Input
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

    # 3. TẠO BỘ LỌC CHỐNG LÓA
    # Chỉ bắt những điểm cực kỳ trắng/sáng (Value > 245, Saturation < 15)
    lower_glare = np.array([0, 0, 245])    
    upper_glare = np.array([180, 15, 255]) 
    
    # Tạo bộ lọc vùng bị lóa (pixel lóa = 255, bình thường = 0)
    glare_filter = cv2.inRange(hsv_img, lower_glare, upper_glare)
    
    # Đảo ngược bộ lọc để GIỮ LẠI vùng bình thường (bỏ vùng lóa)
    mask_giu_lai = cv2.bitwise_not(glare_filter)

    # 4. Chia ảnh làm 3 vùng: Trái, Giữa, Phải
    w_third = width // 3
    feature_vector = []

    for i in range(3):
        start_x = i * w_third
        # Đảm bảo phần Phải lấy sát đến tận mép ảnh
        end_x = width if i == 2 else (i + 1) * w_third
        
        # Cắt vùng tương ứng trên cả ảnh HSV và Mặt nạ
        roi_hsv = hsv_img[:, start_x:end_x]
        roi_mask = mask_giu_lai[:, start_x:end_x]

        # 5. Tính Histogram có ÁP DỤNG BỘ LỌC CHỐNG LÓA
        # Máy sẽ bỏ qua không đếm màu ở những pixel bị lóa
        hist_h = cv2.calcHist([roi_hsv], [0], roi_mask, [bins], [0, 180])
        hist_s = cv2.calcHist([roi_hsv], [1], roi_mask, [bins], [0, 256])
        hist_v = cv2.calcHist([roi_hsv], [2], roi_mask, [bins], [0, 256])

        # Chuẩn hóa (Normalize) histogram
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Nối vector
        feature_vector.extend(hist_h.flatten())
        feature_vector.extend(hist_s.flatten())
        feature_vector.extend(hist_v.flatten())

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