import cv2
import numpy as np
import os

def _to_gray(image_input):
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        return image
    if isinstance(image_input, np.ndarray):
        return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim == 3 else image_input
    return None

def extract_sift_features(image_input):
    gray_img = _to_gray(image_input)
    if gray_img is None: return None, None
    
    sift = cv2.SIFT_create(nfeatures=2000) 
    return sift.detectAndCompute(gray_img, None)

def count_good_matches(kp_test, des_test, kp_template, des_template, ratio_thresh=0.75):

    if (des_test is None or des_template is None or 
        len(des_test) < 4 or len(des_template) < 4):
        return 0
        
    # 1. Khởi tạo bộ so khớp FLANN 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des_test, des_template, k=2)
    except Exception:
        return 0
    
    # 2. Bước lọc 1: Lowe's Ratio Test (Loại bỏ các điểm khớp yếu)
    good_matches_list = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches_list.append(m)
            
    # 3. Bước lọc 2: RANSAC
    # RANSAC giúp loại bỏ các điểm khớp ngẫu nhiên không tuân theo quy luật hình học tờ tiền
    if len(good_matches_list) >= 10: # Nâng ngưỡng lên 10 để lọc nhiễu khắt khe hơn
        src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
        
        # Tìm ma trận đồng dạng và mặt nạ các điểm inliers
        # Ngưỡng ransacReprojThreshold=10.0 cho phép sai số nhẹ trong môi trường thực tế
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        
        if mask is not None:
            # Chỉ đếm những điểm thực sự thuộc về tờ tiền (Inliers)
            return int(np.sum(mask))
            
    return len(good_matches_list)