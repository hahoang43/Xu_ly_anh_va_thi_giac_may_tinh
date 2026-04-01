import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_brightness_adjuster(image_path):
    # 1. INPUT: Đọc ảnh ở dạng Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Không thể đọc được ảnh. Vui lòng kiểm tra đường dẫn.")
        return

    # Tính toán Histogram (số lượng pixel ở mỗi mức sáng từ 0-255)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Tính Hàm phân bố tích lũy (CDF) để lấy các chỉ số thống kê chuẩn xác
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max() # Chuẩn hóa về khoảng [0, 1]

    # Tìm các mốc thống kê (Percentiles) từ Histogram
    p5 = np.argmax(cdf_normalized >= 0.05)  # Mức sáng bao trùm 5% vùng tối nhất
    p50 = np.argmax(cdf_normalized >= 0.50) # Mức sáng trung vị (Median) đại diện cho toàn ảnh
    p95 = np.argmax(cdf_normalized >= 0.95) # Mức sáng bao trùm 95% vùng (loại trừ 5% chói nhất)

    # Tính khoảng động (Dynamic Range) để đo độ tương phản
    dynamic_range = p95 - p5
    
    # Biến lưu trạng thái và ảnh đầu ra
    status = ""
    corrected_img = None

    # Phân loại dựa trên thống kê Histogram
    if dynamic_range < 50:
        status = "Low Contrast"
    elif p50 < 80:
        status = "Too Dark (Underexposed)"
    elif p50 > 170:
        status = "Too Bright (Overexposed)"
    else:
        status = "Normal"

    # 3. AUTO-CORRECT: Áp dụng phép biến đổi tương ứng
    
    if status == "Low Contrast":
        corrected_img = cv2.equalizeHist(img)
        
    elif status == "Too Dark (Underexposed)":
        gamma = 0.5 
        corrected_img = np.array(255 * (img / 255.0) ** gamma, dtype='uint8')
        
    elif status == "Too Bright (Overexposed)":
        gamma = 2.0
        corrected_img = np.array(255 * (img / 255.0) ** gamma, dtype='uint8')
        
    else: 
        corrected_img = img.copy()

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Adaptive Brightness Adjuster\nDetected Status: {status}", fontsize=16, fontweight='bold')

    # Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image (Input)")
    plt.axis('off')

    # Histogram gốc
    plt.subplot(2, 2, 2)
    plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray')
    plt.title("Before Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Ảnh đã hiệu chỉnh
    plt.subplot(2, 2, 3)
    plt.imshow(corrected_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Corrected Image (Output)")
    plt.axis('off')

    # Histogram sau hiệu chỉnh
    plt.subplot(2, 2, 4)
    plt.hist(corrected_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title("After Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

adaptive_brightness_adjuster('low_contrast.jpg') 
