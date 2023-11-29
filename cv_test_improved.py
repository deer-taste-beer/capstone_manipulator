
import cv2
import os
import shutil

# 원본 이미지가 있는 경로
source_dir = '/root/tomato'
# 복사할 이미지가 있는 경로
destination_dir = '/root/tomato/toto'

# 경로 내의 모든 파일 확인
file_list = os.listdir(source_dir)

# 각 파일을 순회하며 빨간색 토마토만을 새로운 경로로 복사
for file_name in file_list:
    file_path = os.path.join(source_dir, file_name)
    if os.path.isfile(file_path):
        # 이미지 파일을 읽기
        img = cv2.imread(file_path)
        if img is not None:
            # BGR 색상 공간을 HSV로 변환
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 초록색 범위 설정 (임의의 초록색 범위 설정)
            lower_green = (35, 50, 50)
            upper_green = (85, 255, 255)
            
            # 초록색 범위에 해당하는 마스크 생성
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
            
            # 초록색 픽셀을 제외한 이미지 생성
            masked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(green_mask))
            
            # 빨간색 범위 설정
            lower_red = (0, 100, 100)
            upper_red = (10, 255, 255)
            
            # 빨간색 범위에 해당하는 마스크 생성
            red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
            
            # 빨간색 픽셀을 식별한 이미지 생성
            red_tomato = cv2.bitwise_and(masked_img, masked_img, mask=red_mask)
            
            # 빨간색 픽셀이 있는 경우 해당 이미지를 복사
            if cv2.countNonZero(red_mask) > 0:
                # 복사할 파일 경로 설정
                destination_file_path = os.path.join(destination_dir, file_name)
                # 이미지 파일을 새로운 경로로 복사
                shutil.copy(file_path, destination_file_path)