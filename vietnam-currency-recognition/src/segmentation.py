import cv2
import numpy as np
import os

from preprocessing import tien_xu_ly_anh 

def sap_xep_toa_do(pts):
    """Sắp xếp 4 đỉnh theo thứ tự: Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    chieu_ngang = np.linalg.norm(rect[1] - rect[0])
    chieu_doc = np.linalg.norm(rect[3] - rect[0])
    
    # Xoay lại nếu tờ tiền bị đặt dọc
    if chieu_doc > chieu_ngang:
        rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")
        
    return rect

def phan_doan_va_nan_chinh(anh_goc, width=800, height=400):
    """Thực hiện Canny, Find Contours và Nắn phẳng trên ảnh gốc"""
    gray = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Canny động
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)

    # Đóng viền
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Tìm viền
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.resize(anh_goc, (width, height))

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    khung_tien = None
    
    h_anh, w_anh = anh_goc.shape[:2]
    dien_tich_toan_anh = h_anh * w_anh

    # Lọc viền theo diện tích và tìm 4 đỉnh
    for c in contours:
        dien_tich_c = cv2.contourArea(c)
        if dien_tich_c < 0.05 * dien_tich_toan_anh or dien_tich_c > 0.98 * dien_tich_toan_anh:
            continue

        chu_vi = cv2.arcLength(c, True)
        for eps in np.linspace(0.01, 0.05, 5):
            approx = cv2.approxPolyDP(c, eps * chu_vi, True)
            if len(approx) == 4:
                khung_tien = approx
                break
                
        if khung_tien is not None:
            break

    # Dự phòng: Khung bao nhỏ nhất
    if khung_tien is None:
        for c in contours:
            if cv2.contourArea(c) > 0.05 * dien_tich_toan_anh:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                khung_tien = np.array(box, dtype=np.int32)
                break

    if khung_tien is None:
        return cv2.resize(anh_goc, (width, height))

    # Nắn phẳng
    pts = khung_tien.reshape(4, 2)
    rect = sap_xep_toa_do(pts)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    anh_nan_phang = cv2.warpPerspective(anh_goc, M, (width, height))
    
    return anh_nan_phang

# =====================================================================
# Main Pipeline
# =====================================================================
if __name__ == "__main__":
    thu_muc_test = "../data/raw" 
    thu_muc_luu = "../data/segmented" 
    
    os.makedirs(thu_muc_luu, exist_ok=True)
    
    if os.path.exists(thu_muc_test):
        for root, dirs, files in os.walk(thu_muc_test):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    duong_dan_goc = os.path.join(root, f)
                    duong_dan_luu = os.path.join(thu_muc_luu, f)
                    
                    try:
                        anh_goc = cv2.imread(duong_dan_goc)
                        if anh_goc is not None:
                            # Bước 1: Gọi hàm phân đoạn ở file này (Cắt & Nắn phẳng)
                            anh_cat_chuan = phan_doan_va_nan_chinh(anh_goc)
                            
                            # Bước 2: Gọi hàm tiền xử lý từ preprocessing.py (CLAHE làm rõ nét)
                            anh_hoan_thien = tien_xu_ly_anh(anh_cat_chuan)
                            
                            # Lưu kết quả cuối cùng
                            cv2.imwrite(duong_dan_luu, anh_hoan_thien)
                    except Exception:
                        pass