# 1. 라이브러리 임포트
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 2. 이미지 불러오기
img1 = cv2.imread('/kaggle/input/image-stitching-samples/1.jpg')
img2 = cv2.imread('/kaggle/input/image-stitching-samples/2.jpg')

# BGR → RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 3. 키포인트 추출 (SIFT 사용)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 키포인트 시각화
img_kp1 = cv2.drawKeypoints(img1, kp1, None)
img_kp2 = cv2.drawKeypoints(img2, kp2, None)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("Image 1 - Keypoints")
plt.imshow(img_kp1)
plt.subplot(1, 2, 2)
plt.title("Image 2 - Keypoints")
plt.imshow(img_kp2)
plt.show()

# 4. 매칭 - KNN + Lowe's Ratio Test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점만 추리기
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 매칭 시각화
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.figure(figsize=(20, 8))
plt.title("Keypoint Matches")
plt.imshow(match_img)
plt.show()

# 5. 호모그래피 계산 (RANSAC 사용)
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 1. 사이즈 확인
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 2. img2의 네 꼭짓점 좌표를 변환
    corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    transformed_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
    
    # 3. 전체 이미지에 필요한 bounding box 계산
    all_corners = np.concatenate((transformed_corners_img2, np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 4. translation matrix로 음수 좌표 보정
    translation = [-x_min, -y_min]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])
    
    # 5. 이미지 2 warp + 이미지 1 붙이기
    result_img = cv2.warpPerspective(img2, T @ H, (x_max - x_min, y_max - y_min))
    result_img[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = img1
    
    # 6. 출력
    plt.figure(figsize=(20,10))
    plt.title("Stitched Image (Planar Projection, with Translation)")
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()


else:
    print("매칭된 점이 부족합니다.")
