import numpy as np
import cv2


# Ảnh grayscale 256x256
def create_blank_image(size=256):
    return np.zeros((size, size), dtype=np.uint8)

# Ảnh gradient ngang
def create_horizontal_gradient(size=256):
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return gradient

# Ảnh checkerboard
def create_checkerboard(size=256, block_size=32):
    img = np.zeros((size, size), dtype=np.uint8)

    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            if (x//block_size + y//block_size) % 2 == 0:
                img[y:y+block_size, x:x+block_size] = 255
    return img

# Ảnh 3: Vòng tròn trắng trên nền đen
def create_circle(size=256, radius=80):
    image = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, radius, 255, -1)
    return image

# (Tùy chọn) Chuyển sang RGB
def convert_to_rgb(gray_image):
    rgb = cv2.merge([gray_image, gray_image, gray_image])
    return rgb

    
# test chạy
if __name__ == "__main__":
    gradient = create_horizontal_gradient()
    checkerboard = create_checkerboard()
    circle = create_circle()
    cv2.imwrite("D:/TGMT/tuan1/gradient.png", gradient)
    cv2.imwrite("D:/TGMT/tuan1/checkerboard.png", checkerboard)
    cv2.imwrite("D:/TGMT/tuan1/circle.png", circle)

    rgb_pattern = cv2.merge([gradient, checkerboard, circle])
    cv2.imwrite("D:/TGMT/tuan1/pattern_rgb.png", rgb_pattern)

    # chuyển RGB (tùy chọn)
    rgb = convert_to_rgb(gradient)
    cv2.imwrite("D:/TGMT/tuan1/gradient_rgb.png", rgb)
    
    print("Đã tạo xong ảnh")