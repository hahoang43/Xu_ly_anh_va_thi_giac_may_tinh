import cv2
import numpy as np
import os

def get_hsv_histogram(image_input, bins=10):
    """
    Hàm nhận vào ảnh tiền đã cắt, chia làm 3 phần và tính HSV Histogram.
    [ĐÃ NÂNG CẤP]: Tích hợp Mặt nạ chống lóa (Glare Mask) để loại bỏ nhiễu đèn flash.
    
    Args:
        image_input: Có thể là đường dẫn ảnh (string) hoặc mảng pixel numpy.
        bins (int): Số lượng bin cho mỗi kênh màu.
        
    Returns:
        numpy.ndarray: Vector đặc trưng 1 chiều.
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
    
    # =========================================================
    # 3. BƯỚC ĐỘT PHÁ: TẠO MẶT NẠ CHỐNG LÓA (GLARE MASK)
    # =========================================================
    # Vùng bị lóa trắng thường có: Value (sáng) > 220, Saturation (bão hòa màu) < 30
    lower_glare = np.array([0, 0, 245])    
    upper_glare = np.array([180, 15, 255]) 
    
    # Tìm các pixel bị lóa (Pixel lóa = 255, Bình thường = 0)
    mask_loa = cv2.inRange(hsv_img, lower_glare, upper_glare)
    
    # Đảo ngược mặt nạ: (Pixel lóa = 0 (Bỏ qua), Bình thường = 255 (Giữ lại để tính màu))
    mask_giu_lai = cv2.bitwise_not(mask_loa)
    # =========================================================

    height, width, _ = hsv_img.shape
    
    # 4. Chia ảnh làm 3 vùng: Trái, Giữa, Phải theo chiều rộng
    w_third = width // 3
    feature_vector = []
    
    for i in range(3):
        start_x = i * w_third
        end_x = width if i == 2 else (i + 1) * w_third
        
        # Cắt ảnh và mặt nạ tương ứng theo từng vùng
        roi_hsv = hsv_img[:, start_x:end_x]
        roi_mask = mask_giu_lai[:, start_x:end_x]
        
        # TÍNH HISTOGRAM CÓ ÁP DỤNG MẶT NẠ (roi_mask)
        # Máy sẽ chỉ đếm màu ở những chỗ KHÔNG BỊ LÓA
        hist = cv2.calcHist([roi_hsv], [0, 1], roi_mask, [bins, bins], [0, 180, 0, 256])
        
        # Chuẩn hóa Vector
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        feature_vector.extend(hist.flatten())

    return np.array(feature_vector, dtype=np.float32)

# ==========================================
# KHU VỰC TEST
# ==========================================
if __name__ == "__main__":
    print("--- Đang kiểm tra hàm tính HSV Histogram (Bản nâng cấp Chống Lóa) ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(current_dir, "..", "data", "raw", "1000", "1000_sau_gap.jpg")
    
    if os.path.exists(test_image_path):
        print(f"[OK] Tìm thấy ảnh thật để test: {test_image_path}")
        try:
            vector_dac_trung = get_hsv_histogram(test_image_path, bins=10)
            print(f"Độ dài vector: {len(vector_dac_trung)} phần tử")
            print("Chạy thành công, tính năng chống lóa đã được kích hoạt!")
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh thật: {e}")
    else:
        print("Vui lòng đặt ảnh vào đúng thư mục để test hàm.")