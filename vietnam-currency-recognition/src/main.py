import cv2
import numpy as np
import os
from segmentation import phan_doan_va_nan_chinh
from preprocessing import tien_xu_ly_anh
from color_features import get_hsv_histogram
from shape_features import extract_sift_features, count_good_matches


# 1. HÀM NẠP MẪU
def nap_mau_tu_dong(thu_muc_raw='data/raw'):
    templates = [] 
    for menh_gia in sorted(os.listdir(thu_muc_raw)):
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia)
        if not os.path.isdir(thu_muc_con):
            continue

        danh_sach_all = [f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        danh_sach_phang = sorted([f for f in danh_sach_all if '_phang_' in f.lower()])
        danh_sach_A = sorted([f for f in danh_sach_all if f.startswith('A_') or f.startswith('a_')])
        danh_sach_khac = sorted([f for f in danh_sach_all if f not in danh_sach_phang and f not in danh_sach_A])
        
        if menh_gia in ['1000', '20000', '200000']:
            gioi_han = 6
        else:
            gioi_han = 4
            
        danh_sach_uu_tien = danh_sach_phang + danh_sach_A + danh_sach_khac

        so_luong_da_nap = 0
        for ten_file in danh_sach_uu_tien:
            if so_luong_da_nap >= gioi_han:
                break 

            duong_dan = os.path.join(thu_muc_con, ten_file)
            img_mau = cv2.imread(duong_dan)
            if img_mau is None:
                continue

            img_mau_cat = phan_doan_va_nan_chinh(img_mau)
            if img_mau_cat is None:
                img_mau_cat = cv2.resize(img_mau, (800, 400)) 

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
                so_luong_da_nap += 1 
                
    print(f"Đã nạp thành công {len(templates)} mẫu chuẩn vào hệ thống!")
    print("Danh sách ảnh mẫu đã nạp:")
    for idx, mau in enumerate(templates, 1):
        print(f" {idx:02d}. {mau['menh_gia']}/{mau['ten_file']}")
    return templates

# 2. NHẬN DIỆN
def nhan_dien_tien(duong_dan_anh_test, templates):
    img_test = cv2.imread(duong_dan_anh_test)
    if img_test is None: return "Lỗi", {}

    img_cat = phan_doan_va_nan_chinh(img_test)
    if img_cat is None:
        img_cat = cv2.resize(img_test, (800, 400))

    img_sach = tien_xu_ly_anh(img_cat)
    if img_sach.ndim == 2:
        img_sach = cv2.cvtColor(img_sach, cv2.COLOR_GRAY2BGR)

    kp_test, des_test = extract_sift_features(img_sach)
    hist_test = get_hsv_histogram(img_cat)

    if des_test is None or len(des_test) < 5: return "Ảnh quá mờ", {}

    bang_diem = []
    for mau in templates:
        np.random.seed(0)
        cv2.setRNGSeed(0)
        try:
            good_matches = count_good_matches(kp_test, des_test, mau['kp'], mau['sift_des'])
        except TypeError:
            good_matches = count_good_matches(des_test, mau['sift_des'])
        
        tong_diem_mau = len(mau['sift_des'])
        ti_le_sift = (good_matches / tong_diem_mau) * 100 if tong_diem_mau > 0 else 0
        diem_hinh_thai = (ti_le_sift + min(good_matches, 100)) / 2
        
        color_sim = cv2.compareHist(hist_test, mau['color_hist'], cv2.HISTCMP_CORREL)
        color_score = max(0, color_sim * 100) 
        
        total_score = (diem_hinh_thai * 0.75) + (color_score * 0.25)
        bang_diem.append({
            'menh_gia': mau['menh_gia'],
            'total_score': total_score
        })
        
    mau_thang_cuoc = max(bang_diem, key=lambda x: x['total_score'])
    
    if mau_thang_cuoc['total_score'] < 12.0:
        return "Không xác định", {}
        
    return mau_thang_cuoc['menh_gia'], {}

# 3. Test toàn bộ dataset
def danh_gia_toan_bo_dataset(thu_muc_raw, templates):
    print(f"\n KIỂM THỬ TỔNG THỂ TRÊN DATASET: {thu_muc_raw}")
    print("-" * 65)
    
    tong_so_anh = 0
    so_cau_dung = 0
    thong_ke = {}
    # cac_menh_gia = ['1000']
    cac_menh_gia = sorted([d for d in os.listdir(thu_muc_raw) if os.path.isdir(os.path.join(thu_muc_raw, d))])
    so_lan_du_doan_la = {mg: 0 for mg in cac_menh_gia}
    so_lan_du_doan_la.update({"Không xác định": 0, "Ảnh quá mờ": 0, "Lỗi": 0})

    for menh_gia_that in cac_menh_gia:
        thu_muc_con = os.path.join(thu_muc_raw, menh_gia_that)
        thong_ke[menh_gia_that] = {'dung': 0, 'tong': 0}

        danh_sach_file = sorted([f for f in os.listdir(thu_muc_con) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        for ten_file in danh_sach_file:
            duong_dan_anh = os.path.join(thu_muc_con, ten_file)
            tong_so_anh += 1
            thong_ke[menh_gia_that]['tong'] += 1

            ket_qua_doan, _ = nhan_dien_tien(duong_dan_anh, templates)

            if ket_qua_doan in so_lan_du_doan_la:
                so_lan_du_doan_la[ket_qua_doan] += 1
            else:
                so_lan_du_doan_la[ket_qua_doan] = 1

            if ket_qua_doan == menh_gia_that:
                so_cau_dung += 1
                thong_ke[menh_gia_that]['dung'] += 1
                status = "Đúng"
            else:
                status = f"SAI ({ket_qua_doan})"
            print(f"[{tong_so_anh:03d}] {ten_file:<25} -> {status}")

    # IN BẢNG BÁO CÁO 
    print("\n" + "="*80)
    print(f"ĐỘ CHÍNH XÁC TỔNG (OVERALL ACCURACY): {(so_cau_dung / tong_so_anh) * 100:.2f}%")
    print("="*80)
    print(f"{'MỆNH GIÁ':<10} | {'ĐÚNG/TỔNG':<10} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10}")
    print("-" * 80)

    for mg in cac_menh_gia:
        tong_anh = thong_ke[mg]['tong']
        so_lan_dung = thong_ke[mg]['dung']
        tong_du_doan = so_lan_du_doan_la.get(mg, 0)

        if tong_anh > 0:
            acc_percent = (so_lan_dung / tong_anh) * 100
            recall = so_lan_dung / tong_anh
            precision = so_lan_dung / tong_du_doan if tong_du_doan > 0 else 0.0

            print(f" {mg:<9} | {so_lan_dung:02d}/{tong_anh:02d}      | {acc_percent:6.2f}%   | {precision:8.4f} | {recall:8.4f}")
    
    print("="*80)

if __name__ == "__main__":
    thu_muc_du_lieu = 'data/raw' 
    if not os.path.exists(thu_muc_du_lieu):
        thu_muc_du_lieu = '../data/raw'
        
    bo_mau_chuan = nap_mau_tu_dong(thu_muc_du_lieu) 
    
    if len(bo_mau_chuan) > 0:
        danh_gia_toan_bo_dataset(thu_muc_du_lieu, bo_mau_chuan)