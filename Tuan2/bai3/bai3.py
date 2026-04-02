import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Phân tích ảnh
def analyze_image(img):
    total_pixels = img.size
    # Đếm số pixel bị nhiễu sát 0 và 255 
    num_outliers = np.sum((img <= 2) | (img >= 253))
    outlier_ratio = num_outliers / total_pixels
    std = np.std(img)
    return {"outlier_ratio": outlier_ratio, "std": std}

# Phát hiện loại nhiễu
def detect_noise(stats):
    if stats["outlier_ratio"] > 0.03: 
        return "salt_pepper"
    elif stats["std"] > 15: 
        return "gaussian"
    return "other"

#  Hàm hiển thị GỘP CẢ BIỂU ĐỒ VÀ ẢNH
def show_all_in_one(noisy, filtered, report):
    plt.figure(figsize=(12, 8))
    
    # Hiển thị ảnh nhiễu
    plt.subplot(2, 2, 1)
    plt.imshow(noisy, cmap='gray')
    plt.title(f"Noisy Image ({report['Detected Noise']})")
    plt.axis("off")

    # Hiển thị Histogram ảnh nhiễu 
    plt.subplot(2, 2, 2)
    plt.hist(noisy.ravel(), bins=256, color='Blue')
    plt.title("Histogram - Noisy ")

    # Hiển thị ảnh sau lọc
    plt.subplot(2, 2, 3)
    plt.imshow(filtered, cmap='gray')
    plt.title(f"Filtered Result ({report['Filter Applied']})")
    plt.axis("off")

    # Hiển thị Histogram sau lọc
    plt.subplot(2, 2, 4)
    plt.hist(filtered.ravel(), bins=256, color='blue')
    plt.title("Histogram - Filtered (Smoothed)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    noisy_path = r"noisy.png"# ảnh bị nhiễu
    
    if not os.path.exists(noisy_path):
        print("Lỗi: Không tìm thấy file ảnh!")
    else:
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        stats = analyze_image(noisy_img)
        noise_type = detect_noise(stats)

        # Chọn bộ lọc phù hợp
        if noise_type == "salt_pepper":
            filtered_img = cv2.medianBlur(noisy_img, 5) 
            filter_used = "Median Filter"
        elif noise_type == "gaussian":
            filtered_img = cv2.bilateralFilter(noisy_img, 9, 75, 75) 
            filter_used = "Bilateral Filter"
        else:
            filtered_img = noisy_img.copy()
            filter_used = "None"

        report = {"Detected Noise": noise_type, "Filter Applied": filter_used}

        # Kết quả
        print("\nKết quả: ")
        print(f"Loại nhiễu : {noise_type}")
        print(f"Bộ lọc áp dụng : {filter_used}")
        print(f"Tỷ lệ điểm ngoại lai  : {stats['outlier_ratio']:.4f}")
        print(f"Độ lệch chuẩn (Std)   : {stats['std']:.2f}")

        # Hiển thị kết quả
        show_all_in_one(noisy_img, filtered_img, report)