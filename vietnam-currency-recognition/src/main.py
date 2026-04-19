import cv2
import numpy as np
import os

# Import các module 
from segmentation import phan_doan_va_nan_chinh
from preprocessing import tien_xu_ly_anh
from color_features import get_hsv_histogram
from shape_features import extract_sift_features, count_good_matches

# 1. HÀM TỰ ĐỘNG NẠP MẪU 
def nap_mau_tu_dong(thu_muc_raw='data/raw'):
    templates = [] 
    for menh_gia in sorted(os.listdir(thu_muc_raw)):
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia)
        if not os.path.isdir(thu_muc_con): continue
        # nạp mẫu ảnh phẳng
        danh_sach_phang = sorted([f for f in os.listdir(thu_muc_con) if "_phang_" in f.lower()])
        
        # Nếu không tìm thấy ảnh phẳng, mới lấy các file khác
        if len(danh_sach_phang) < 2:
            danh_sach_all = sorted([f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.png'))])
            anh_duoc_chon = danh_sach_all[:2]
        else:
            anh_duoc_chon = danh_sach_phang[:2]
            
        for ten_file in anh_duoc_chon:
            
            duong_dan = os.path.join(thu_muc_con, ten_file)
            img_mau = cv2.imread(duong_dan)
            if img_mau is None: continue
            
            # Xử lý ảnh mẫu để đưa vào bộ nhớ RAM
            img_mau_cat = phan_doan_va_nan_chinh(img_mau)
            img_mau_sach = tien_xu_ly_anh(img_mau_cat)
            
            hist_mau = get_hsv_histogram(img_mau_cat)
            kp, des = extract_sift_features(img_mau_sach)
            
            if des is not None:
                templates.append({
                    'menh_gia': menh_gia,
                    'ten_file': ten_file,
                    'kp': kp,
                    'sift_des': des,
                    'color_hist': hist_mau
                })
                
    print(f"Da nap thanh cong {len(templates)} mau chuan vao he thong!")
    # Hien thi ten cac anh mau da nap
    print("Danh sach anh mau da nap:")
    for idx, mau in enumerate(templates, 1):
        print(f" {idx:02d}. {mau['menh_gia']}/{mau['ten_file']}")
    return templates

# 2. NHẬN DIỆN 
def nhan_dien_tien(duong_dan_anh_test, templates):
    img_test = cv2.imread(duong_dan_anh_test)
    if img_test is None: return "Lỗi", {}

    # Bước 1: Tiền xử lý trực tiếp trên RAM
    img_cat = phan_doan_va_nan_chinh(img_test)
    img_sach = tien_xu_ly_anh(img_cat)

    # Bước 2: Trích đặc trưng của ảnh đang test
    kp_test, des_test = extract_sift_features(img_sach)
    hist_test = get_hsv_histogram(img_cat)

    if des_test is None or len(des_test) < 5: return "Ảnh quá mờ", {}

    bang_diem = []

    # Bước 3: Đối sánh với từng mẫu trong từ điển
    for mau in templates:
        # --- Chấm điểm SIFT (Hình thái) ---
        good_matches = count_good_matches(kp_test, des_test, mau['kp'], mau['sift_des'])
        
        # Công thức chuẩn hóa %: Tránh việc các tờ ít Keypoints (như 10k) chiếm ưu thế
        tong_diem_mau = len(mau['sift_des'])
        ti_le_sift = (good_matches / tong_diem_mau) * 100 if tong_diem_mau > 0 else 0
        
        # Kết hợp Tỷ lệ % và Số lượng tuyệt đối
        # Capped tuyệt đối ở 100 để tờ 500k không lấn át các tờ khác
        diem_hinh_thai = (ti_le_sift + min(good_matches, 100)) / 2
                
        # --- Chấm điểm Màu sắc (HSV) ---
        color_sim = cv2.compareHist(hist_test, mau['color_hist'], cv2.HISTCMP_CORREL)
        color_score = max(0, color_sim * 100) 
        
        # --- TỔNG ĐIỂM TRỌNG SỐ ---
        # 60% Hình dạng + 40% Màu sắc (Tăng màu sắc để phân biệt 1k và 10k tốt hơn)
        total_score = (diem_hinh_thai * 0.75) + (color_score * 0.25)
        
        bang_diem.append({
            'menh_gia': mau['menh_gia'],
            'total_score': total_score
        })
        
    # Chọn mệnh giá có điểm cao nhất
    mau_thang_cuoc = max(bang_diem, key=lambda x: x['total_score'])
    
    # Ngưỡng từ chối (Nếu điểm quá thấp thì không kết luận)
    if mau_thang_cuoc['total_score'] < 12.0:
        return "Không xác định", {}
        
    return mau_thang_cuoc['menh_gia'], {}

# 3. HÀM QUÉT TOÀN BỘ DATASET ĐỂ TÍNH ACCURACY
def danh_gia_toan_bo_dataset(thu_muc_raw, templates):
    print(f"\nKIEM THU TONG THE TREN DATASET: {thu_muc_raw}")
    print("-" * 65)
    
    tong_so_anh = 0
    so_cau_dung = 0
    thong_ke = {}

    for menh_gia_that in sorted(os.listdir(thu_muc_raw)):
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia_that)
        if not os.path.isdir(thu_muc_con): continue

        thong_ke[menh_gia_that] = {'dung': 0, 'tong': 0}

        for ten_file in sorted(os.listdir(thu_muc_con)):
            if not ten_file.lower().endswith(('.jpg', '.png', '.jpeg')): continue

            duong_dan_anh = os.path.join(thu_muc_con, ten_file)
            tong_so_anh += 1
            thong_ke[menh_gia_that]['tong'] += 1

            ket_qua_doan, _ = nhan_dien_tien(duong_dan_anh, templates)

            if ket_qua_doan == menh_gia_that:
                so_cau_dung += 1
                thong_ke[menh_gia_that]['dung'] += 1
                status = "Dung"
            else:
                status = f"SAI ({ket_qua_doan})"

            print(f"[{tong_so_anh:03d}] {ten_file:<25} -> {status}")

    # IN BÁO CÁO CUỐI CÙNG
    print("\n" + "="*60)
    print(f"DO CHINH XAC TONG (ACCURACY): {(so_cau_dung / tong_so_anh) * 100:.2f}%")
    print("="*60)
    for mg, kq in thong_ke.items():
        if kq['tong'] > 0:
            acc = (kq['dung'] / kq['tong']) * 100
            print(f" {mg:<7}: {acc:5.1f}% ({kq['dung']:02d}/{kq['tong']:02d} anh)")

if __name__ == "__main__":
    thu_muc_du_lieu = 'data/raw' 
    
    # Tự động nạp mẫu từ folder raw
    bo_mau_chuan = nap_mau_tu_dong(thu_muc_du_lieu) 
    
    # Chạy kiểm tra toàn bộ 300+ ảnh
    if len(bo_mau_chuan) > 0:
        danh_gia_toan_bo_dataset(thu_muc_du_lieu, bo_mau_chuan)
