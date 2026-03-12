import cv2
import numpy as np

# Hàm clip giá trị về khoảng [0,255] và chuyển sang uint8
def clip_uint8(arr):
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


# 1. Đọc ảnh grayscale
gray = cv2.imread("gray_manual.png", cv2.IMREAD_GRAYSCALE)

if gray is None:
    print("Không đọc được ảnh")
    exit()

print("Shape ảnh grayscale:", gray.shape)

# Lấy giá trị pixel ví dụ
print("Pixel gốc [100,100]:", gray[100, 100])


# 2. Tạo ảnh tối hơn
gray_dark = clip_uint8(gray - 50)

# 3. Tạo ảnh sáng hơn
gray_bright = clip_uint8(gray + 50)

# 4. Tăng tương phản
alpha = 1.5
gray_contrast = clip_uint8(gray * alpha)


# 5. Threshold nhị phân (tự cài đặt)
T = 128

binary = np.zeros_like(gray)
binary[gray >= T] = 255


# 6. In giá trị pixel sau khi xử lý
print("Pixel tối hơn [100,100]:", gray_dark[100,100])
print("Pixel sáng hơn [100,100]:", gray_bright[100,100])
print("Pixel tăng tương phản [100,100]:", gray_contrast[100,100])
print("Pixel binary [100,100]:", binary[100,100])


# 7. Lưu ảnh
cv2.imwrite("gray_dark.png", gray_dark)
cv2.imwrite("gray_bright.png", gray_bright)
cv2.imwrite("gray_contrast.png", gray_contrast)
cv2.imwrite("binary.png", binary)

print("Đã lưu tất cả ảnh.")