import cv2
import numpy as np
import shutil
import os

# 이미지 디렉토리 경로 설정
image_directory = '/root/tomato'
ripe_directory = '/root/tomato/toto'
unripe_directory = '/root/tomato/grgr'

# 경로가 없으면 생성
os.makedirs(ripe_directory, exist_ok=True)
os.makedirs(unripe_directory, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = os.listdir(image_directory)

for img_file in image_files:
    # 이미지 확장자 확인
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

        # 이미지 읽기
        image = cv2.imread(os.path.join(image_directory, img_file))

        # 이미지를 HSV(Hue, Saturation, Value)로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 익지 않은 토마토를 위한 색 범위 설정 (녹색 계열)
        lower_green = np.array([35, 50, 50])  # 색 범위에 따라 조정 가능
        upper_green = np.array([90, 255, 255])  # 색 범위에 따라 조정 가능

        # 빨간색 토마토를 위한 색 범위 설정 (빨간 계열)
        lower_red = np.array([0, 50, 50])  # 색 범위에 따라 조정 가능
        upper_red = np.array([10, 255, 255])  # 색 범위에 따라 조정 가능

        # 각 색상에 해당하는 마스크 생성
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # 토마토 영역의 마스크 생성 (녹색과 빨간색을 합침)
        tomato_mask = cv2.bitwise_or(mask_green, mask_red)

        # 이미지 전처리 (노이즈 제거)
        kernel = np.ones((5, 5), np.uint8)
        tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel)

        # 컨투어(윤곽선) 찾기
        contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어가 비어 있는 경우 처리
        if not contours:
            continue

        # 컨투어 중 가장 큰 영역 찾기
        max_contour = max(contours, key=cv2.contourArea)

        # 컨투어로부터 사각형 영역 획득
        x, y, w, h = cv2.boundingRect(max_contour)

        # 토마토 영역 추출
        tomato_roi = image[y:y+h, x:x+w]

        # 토마토 영역을 다시 HSV로 변환
        hsv_roi = cv2.cvtColor(tomato_roi, cv2.COLOR_BGR2HSV)

        # 토마토 영역의 색상 분석 (빨간색 마스크 적용)
        mask_roi_red = cv2.inRange(hsv_roi, lower_red, upper_red)
        contours_roi_red, _ = cv2.findContours(mask_roi_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 익은 토마토로 판별 후 해당 디렉토리로 복사
        if len(contours_roi_red) > 0:
            shutil.copy(os.path.join(image_directory, img_file), os.path.join(ripe_directory, img_file))
        else:
            shutil.copy(os.path.join(image_directory, img_file), os.path.join(unripe_directory, img_file))

print("이미지 분류가 완료되었습니다.")
