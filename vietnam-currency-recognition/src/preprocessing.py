import cv2
import numpy as np
import os

def tien_xu_ly_anh(input_data):
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"[Lỗi Preprocessing] Không tìm thấy ảnh tại: {input_data}")
        img = cv2.imread(input_data)
    elif isinstance(input_data, np.ndarray):
        img = input_data.copy()
    else:
        raise ValueError("[Lỗi Preprocessing] Đầu vào phải là đường dẫn (string) hoặc ảnh (numpy.ndarray)")

    if img is None:
        raise ValueError("[Lỗi Preprocessing] Không thể đọc được dữ liệu ảnh. Hãy kiểm tra lại file.")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    anh_ro_net = clahe.apply(blurred)

    return anh_ro_net

# Test 
#  if __name__ == "__main__":
    thu_muc_test = "../data/raw" 
    
    if not os.path.exists(thu_muc_test):
        print(f" Lỗi: Chưa có thư mục '{thu_muc_test}'. Hãy tạo và bỏ các thư mục ảnh con vào!")
        exit()
        
    danh_sach_anh = []
    for root, dirs, files in os.walk(thu_muc_test):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                danh_sach_anh.append(os.path.join(root, f))
    
    if len(danh_sach_anh) == 0:
        print(f" Lỗi: Thư mục '{thu_muc_test}' và các thư mục con đang trống!")
        exit()

    print(f"Bắt đầu test CLAHE trên {len(danh_sach_anh)} bức ảnh...")
    print("HƯỚNG DẪN: Bấm phím BẤT KỲ để chuyển sang ảnh tiếp theo. Bấm phím 'q' hoặc 'ESC' để thoát.\n")

    for duong_dan_anh in danh_sach_anh:
        print(f"Đang xử lý: {duong_dan_anh}")
        try:
            anh_ket_qua = tien_xu_ly_anh(duong_dan_anh)
            anh_goc = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
            
            h_goc, w_goc = anh_goc.shape[:2]
            anh_goc_resize = cv2.resize(anh_goc, (int(w_goc * 500 / h_goc), 500))
            anh_ket_qua_resize = cv2.resize(anh_ket_qua, (int(w_goc * 500 / h_goc), 500))
            
            anh_so_sanh = np.hstack((anh_goc_resize, anh_ket_qua_resize))
            cv2.imshow("Trai: Anh Goc | Phai: Anh sau khi qua CLAHE", anh_so_sanh)
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == ord('q'): 
                print(" thoát quá trình test.")
                break
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {duong_dan_anh}: {e}")

    cv2.destroyAllWindows()
    print(" Đã test xong toàn bộ ảnh!")