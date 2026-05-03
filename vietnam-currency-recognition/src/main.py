import cv2
import numpy as np
import os

# Import các module chuyên gia của nhóm
from segmentation import phan_doan_va_nan_chinh
from preprocessing import tien_xu_ly_anh
from color_features import get_hsv_histogram
from shape_features import extract_sift_features, count_good_matches

# ==============================================================
# 1. HÀM TỰ ĐỘNG NẠP MẪU (Nạp mẫu có chọn lọc)
# ==============================================================
def nap_mau_tu_dong(thu_muc_raw='data/raw'):
    templates = [] 
    for menh_gia in sorted(os.listdir(thu_muc_raw)):
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia)
        if not os.path.isdir(thu_muc_con):
            continue

        # Ưu tiên lấy các file _phang_ trước
        danh_sach_phang = sorted([f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_phang_' in f.lower()])
        # Sau đó lấy thêm các file còn lại
        danh_sach_khac = sorted([f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in danh_sach_phang])
        
        # [CHIẾN THUẬT MỚI]: Nâng ngưỡng cho 3 mệnh giá yếu
        if menh_gia in ['10000', '50000', '200000']:
            gioi_han = 6
        else:
            gioi_han = 4
            
        # Gộp lại, ưu tiên _phang_ lên đầu, lấy theo giới hạn đã thiết lập
        danh_sach_anh = (danh_sach_phang + danh_sach_khac)[:gioi_han]

        for ten_file in danh_sach_anh:
            duong_dan = os.path.join(thu_muc_con, ten_file)
            img_mau = cv2.imread(duong_dan)
            if img_mau is None:
                continue

            # Xử lý ảnh mẫu để đưa vào bộ nhớ RAM
            img_mau_cat = phan_doan_va_nan_chinh(img_mau)
            if img_mau_cat is None:
                img_mau_cat = cv2.resize(img_mau, (800, 400)) # Fallback nếu không cắt được

            img_mau_sach = tien_xu_ly_anh(img_mau_cat)
            if img_mau_sach.ndim == 2:
                img_mau_sach = cv2.cvtColor(img_mau_sach, cv2.COLOR_GRAY2BGR)

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
                
    print(f"✅ Đã nạp thành công {len(templates)} mẫu chuẩn vào hệ thống!")
    print("Danh sách ảnh mẫu đã nạp:")
    for idx, mau in enumerate(templates, 1):
        print(f" {idx:02d}. {mau['menh_gia']}/{mau['ten_file']}")
    return templates
# ==============================================================
# 2. BỘ NÃO NHẬN DIỆN (CÔNG THỨC HYBRID SCORING MỚI)
# ==============================================================
def nhan_dien_tien(duong_dan_anh_test, templates):
    img_test = cv2.imread(duong_dan_anh_test)
    if img_test is None: return "Lỗi", {}

    # Bước 1: Tiền xử lý trực tiếp trên RAM
    # [CHỐT AN TOÀN 2]: Fallback nếu cắt ảnh test bị fail
    img_cat = phan_doan_va_nan_chinh(img_test)
    if img_cat is None:
        img_cat = cv2.resize(img_test, (800, 400))

    img_sach = tien_xu_ly_anh(img_cat)
    if img_sach.ndim == 2:
        img_sach = cv2.cvtColor(img_sach, cv2.COLOR_GRAY2BGR)

    # Bước 2: Trích đặc trưng của ảnh đang test
    kp_test, des_test = extract_sift_features(img_sach)
    hist_test = get_hsv_histogram(img_cat)

    if des_test is None or len(des_test) < 5: return "Ảnh quá mờ", {}

    bang_diem = []

    # Bước 3: Đối sánh với từng mẫu trong từ điển
    for mau in templates:
        # --- Chấm điểm SIFT (Hình thái) ---
        try:
            good_matches = count_good_matches(kp_test, des_test, mau['kp'], mau['sift_des'])
        except TypeError:
            good_matches = count_good_matches(des_test, mau['sift_des'])
        
        # Công thức chuẩn hóa %
        tong_diem_mau = len(mau['sift_des'])
        ti_le_sift = (good_matches / tong_diem_mau) * 100 if tong_diem_mau > 0 else 0
        
        # ĐIỂM LAI (HYBRID): Kết hợp Tỷ lệ % và Số lượng tuyệt đối
        diem_hinh_thai = (ti_le_sift + min(good_matches, 100)) / 2
                
        # --- Chấm điểm Màu sắc (HSV) ---
        color_sim = cv2.compareHist(hist_test, mau['color_hist'], cv2.HISTCMP_CORREL)
        color_score = max(0, color_sim * 100) 
        
        # --- TỔNG ĐIỂM TRỌNG SỐ ---
        # Chuẩn hóa: 75% Hình thái SIFT + 25% Màu sắc HSV (Khớp với báo cáo Chương 3)
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

# ==============================================================
# 3. HÀM QUÉT TOÀN BỘ DATASET ĐỂ TÍNH ACCURACY
# ==============================================================
# def danh_gia_toan_bo_dataset(thu_muc_raw, templates):
    print(f"\n🚀 KIỂM THỬ TỔNG THỂ TRÊN DATASET: {thu_muc_raw}")
    print("-" * 65)
    
    tong_so_anh = 0
    so_cau_dung = 0
    thong_ke = {}

    for menh_gia_that in sorted(os.listdir(thu_muc_raw)):
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia_that)
        if not os.path.isdir(thu_muc_con):
            continue

        thong_ke[menh_gia_that] = {'dung': 0, 'tong': 0}

        # Quét toàn bộ ảnh hợp lệ trong thư mục mệnh giá
        danh_sach_file = sorted([f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        for ten_file in danh_sach_file:
            duong_dan_anh = os.path.join(thu_muc_con, ten_file)
            tong_so_anh += 1
            thong_ke[menh_gia_that]['tong'] += 1

            ket_qua_doan, _ = nhan_dien_tien(duong_dan_anh, templates)

            if ket_qua_doan == menh_gia_that:
                so_cau_dung += 1
                thong_ke[menh_gia_that]['dung'] += 1
                status = "✅ Đúng"
            else:
                status = f"❌ SAI ({ket_qua_doan})"

            print(f"[{tong_so_anh:03d}] {ten_file:<25} -> {status}")

    # IN BÁO CÁO CUỐI CÙNG
    print("\n" + "="*60)
    print(f"🎯 ĐỘ CHÍNH XÁC TỔNG (ACCURACY): {(so_cau_dung / tong_so_anh) * 100:.2f}%")
    print("="*60)
    for mg, kq in thong_ke.items():
        if kq['tong'] > 0:
            acc = (kq['dung'] / kq['tong']) * 100
            print(f" 💵 {mg:<7}: {acc:5.1f}% ({kq['dung']:02d}/{kq['tong']:02d} ảnh)")

# if __name__ == "__main__":
    # Đã sửa lại đường dẫn thư mục chuẩn (Tùy thuộc vào việc Hà mở Terminal ở thư mục gốc hay src)
    thu_muc_du_lieu = 'data/raw' 
    if not os.path.exists(thu_muc_du_lieu):
        thu_muc_du_lieu = '../data/raw'
        
    # Tự động nạp mẫu từ folder raw
    bo_mau_chuan = nap_mau_tu_dong(thu_muc_du_lieu) 
    
    # Nếu muốn test riêng 1 mệnh giá, ví dụ '1000', hãy gọi hàm test_anh_menh_gia
    if len(bo_mau_chuan) > 0:
        test_anh_menh_gia(thu_muc_du_lieu, '1000', bo_mau_chuan)
        # Nếu muốn test toàn bộ, bỏ comment dòng dưới:
        danh_gia_toan_bo_dataset(thu_muc_du_lieu, bo_mau_chuan)