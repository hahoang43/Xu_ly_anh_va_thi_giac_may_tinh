import cv2
import numpy as np
import time
from scipy.signal import fftconvolve

# Tạo kernel Gaussian
def gaussian_kernel(ksize, sigma=0):
    if sigma == 0:
        sigma = ksize / 6.0
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# Spatial convolution (dùng OpenCV)
def spatial_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# FFT convolution
def fft_blur(img, kernel):
    result = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):  # xử lý từng kênh màu
        result[:, :, c] = fftconvolve(img[:, :, c], kernel, mode='same')

    return np.clip(result, 0, 255).astype(np.uint8)

# Hàm chọn phương pháp nhanh nhất
def auto_gaussian_blur(image, ksize):
    kernel = gaussian_kernel(ksize)

    # Crop 100x100
    h, w = image.shape[:2]
    crop = image[0:min(100, h), 0:min(100, w)]

    # --- Test Spatial ---
    start = time.time()
    _ = spatial_blur(crop, ksize)
    time_spatial = time.time() - start

    # --- Test FFT ---
    start = time.time()
    _ = fft_blur(crop, kernel)
    time_fft = time.time() - start

    print(f"Time Spatial: {time_spatial:.6f}s")
    print(f"Time FFT: {time_fft:.6f}s")

    # --- Chọn method ---
    if time_spatial < time_fft:
        print("Chọn: Spatial Convolution")
        return spatial_blur(image, ksize)
    else:
        print("Chọn: FFT Convolution")
        return fft_blur(image, kernel)


# ================= MAIN =================
if __name__ == "__main__":
    img = cv2.imread("image1.jpg")

    ksize = 1  # bạn có thể thay đổi 15

    result = auto_gaussian_blur(img, ksize)

    cv2.imshow("Original", img)
    cv2.imshow("Blurred", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output.jpg", result)