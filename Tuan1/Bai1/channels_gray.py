import cv2
import numpy as np

# Đọc ảnh màu
img = cv2.imread('photo.jpg')

if img is None:
    print("Lỗi: Không tìm thấy file photo.jpg. Vui lòng kiểm tra lại đường dẫn.")
    exit()

#Tách kênh bằng NumPy indexing
B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

# Lưu từng kênh màu
cv2.imwrite('blue.png', B)
cv2.imwrite('green.png', G)
cv2.imwrite('red.png', R)

# Tự cài đặt chuyển RGB -> grayscale
R_f = R.astype(np.float32)
G_f = G.astype(np.float32)
B_f = B.astype(np.float32)

# Option 1: Trung bình cộng
gray_opt1 = (R_f + G_f + B_f) / 3
gray_opt1 = np.clip(gray_opt1, 0, 255).astype(np.uint8)

# Option 2: Trọng số 
gray_opt2 = 0.299 * R_f + 0.587 * G_f + 0.114 * B_f
gray_opt2 = np.clip(gray_opt2, 0, 255).astype(np.uint8)

# Lưu ảnh grayscale 
cv2.imwrite('gray_manual.png', gray_opt2)

# In kết quả
print("--- KẾT QUẢ SO SÁNH SHAPE ---")
print(f"Shape của ảnh màu gốc: {img.shape}")
print(f"Shape của kênh Blue  : {B.shape}")
print(f"Shape của kênh Green : {G.shape}")
print(f"Shape của kênh Red   : {R.shape}")
print(f"Shape của ảnh Grayscale: {gray_opt2.shape}")