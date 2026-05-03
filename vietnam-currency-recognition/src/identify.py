import cv2
import os

# ==============================================================
# TỐI ƯU 1: ÉP KÍCH THƯỚC ẢNH NHỎ LẠI ĐỂ SIFT CHẠY NHANH
# ==============================================================
def resize_anh(img, max_width=600):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

# ==============================================================
# TỐI ƯU 2: TÍNH SẴN ĐẶC TRƯNG CỦA 9 TỜ TIỀN MẪU (CHỈ CHẠY 1 LẦN)
# ==============================================================
def tai_du_lieu_mau(thu_muc_mau='data/raw'):
    print("⏳ Đang nạp hệ thống mẫu SIFT (Chỉ chạy 1 lần duy nhất)...")
    sift = cv2.SIFT_create()
    templates = {}
    
    for ten_file in os.listdir(thu_muc_mau):
        if not ten_file.endswith(('.jpg', '.png')): continue
        menh_gia = ten_file.split('_')[0]
        duong_dan = os.path.join(thu_muc_mau, ten_file)
        
        # Đọc và resize ảnh mẫu
        img_mau = cv2.imread(duong_dan, cv2.IMREAD_GRAYSCALE)
        if img_mau is None: continue
        img_mau = resize_anh(img_mau)
        
        # Tính SIFT
        kp, des = sift.detectAndCompute(img_mau, None)
        if des is not None:
            templates[menh_gia] = des # Chỉ cần lưu Descriptors là đủ để so sánh
            
    print("✅ Nạp xong hệ thống mẫu!")
    return templates

# Khởi tạo biến toàn cục (Chạy ngay khi mở Streamlit)
MAU_SIFT_DA_LUU = tai_du_lieu_mau()

# ==============================================================
# HÀM NHẬN DIỆN CHÍNH (Đã dùng bộ so khớp FLANN siêu tốc)
# ==============================================================
def nhan_dien_tien_sieu_nhanh(duong_dan_anh_test):
    img_test = cv2.imread(duong_dan_anh_test, cv2.IMREAD_GRAYSCALE)
    if img_test is None: return "Lỗi ảnh"
    
    # Resize ảnh test
    img_test = resize_anh(img_test)

    # Tính SIFT cho ảnh test
    sift = cv2.SIFT_create()
    kp_test, des_test = sift.detectAndCompute(img_test, None)
    if des_test is None: return "Ảnh quá mờ"

    # TỐI ƯU 3: DÙNG FLANN MATCHER THAY CHO BRUTE-FORCE
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    ket_qua_cham_diem = {}

    # So sánh với mảng mẫu đã lưu sẵn trên RAM
    for menh_gia, des_mau in MAU_SIFT_DA_LUU.items():
        # Tìm 2 điểm gần nhất
        matches = flann.knnMatch(des_test, des_mau, k=2)
        
        # Lowe's Ratio Test
        good_matches = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches += 1
                
        ket_qua_cham_diem[menh_gia] = good_matches
        
    # Tìm kết quả cao nhất
    menh_gia_thang_cuoc = max(ket_qua_cham_diem, key=ket_qua_cham_diem.get)
    diem_cao_nhat = ket_qua_cham_diem[menh_gia_thang_cuoc]
    
    if diem_cao_nhat < 15:
        return "Không nhận diện được", ket_qua_cham_diem
        
    return menh_gia_thang_cuoc, ket_qua_cham_diem

# --- TEST THỬ ---
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    import os
    test_img_rel = "data/raw/50000/50000_truoc_phang_01.jpg"
    test_img_abs = os.path.abspath(test_img_rel)
    print(f"🖼️ Đường dẫn tuyệt đối kiểm tra: {test_img_abs}")
    print(f"🗂️ Ảnh tồn tại? {os.path.exists(test_img_abs)}")
    result = nhan_dien_tien_sieu_nhanh(test_img_abs)
    end_time = time.time()
    if isinstance(result, tuple):
        ket_qua, chi_tiet = result
        print(f"🎉 KẾT QUẢ: {ket_qua} VNĐ")
        print(f"📊 Bảng điểm chi tiết: {chi_tiet}")
    else:
        print("❌ Lỗi:", result)
    print(f"⏱️ Thời gian xử lý: {end_time - start_time:.2f} giây")