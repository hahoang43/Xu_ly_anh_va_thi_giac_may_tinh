import cv2
import os

# ===== Hàm tính dung lượng file (KB) =====
def file_size_kb(path):
    size = os.path.getsize(path)  # kích thước tính bằng byte
    return round(size / 1024, 2)


# ===== Đường dẫn ảnh gốc =====
input_path = r"D:\thigiacmt\E-1\Xu_ly_anh_va_thi_giac_may_tinh\image.jpg"

# ===== Đọc ảnh =====
img = cv2.imread(input_path)

if img is None:
    print("Không đọc được ảnh! Kiểm tra lại đường dẫn.")
    exit()

print("=== THÔNG SỐ ẢNH GỐC ===")
print("Shape:", img.shape)
print("Dung lượng:", file_size_kb(input_path), "KB")


# ===== Các mức giảm độ phân giải =====
sizes = [
    (512, 512),
    (256, 256),
    (128, 128),
    (64, 64)
]

# ===== Resampling =====
for w, h in sizes:

    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    output_name = f"output_{w}x{h}.jpg"

    cv2.imwrite(output_name, resized)

    print("\nẢnh:", output_name)
    print("Shape:", resized.shape)
    print("Dung lượng:", file_size_kb(output_name), "KB")
    