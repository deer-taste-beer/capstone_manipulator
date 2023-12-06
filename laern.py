import os
import cv2
import numpy as np
import pickle

# 라벨과 이미지가 저장된 폴더 경로
label_folder = r'C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\target' 
image_folder = r'C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\따로 실험용12_06'

# 학습 데이터를 저장할 리스트
data = []

# 색상을 기반으로 토마토가 익었는지 판단하는 함수
def determine_ripeness(image):
    # 이미지를 HSV 색공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 빨간색 범위인데 수정하는게 좋을 듯
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 이미지에서 빨간색 마스크 생성
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 마스크를 적용하여 빨간색 영역만 추출
    red_area = cv2.bitwise_and(image, image, mask=mask)

    # 빨간색 영역의 비율 계산
    red_ratio = np.sum(mask) / (image.shape[0] * image.shape[1])

    # 빨간색 영역의 비율이 0.5 이상이면 토마토가 익었다고 판단 이 부분도 추후 50%이상보다 더 빨간 부분이 많아야 익었다고 바꿔줄 수 있음
    return red_ratio > 0.5

# 폴더의 모든 파일을 순회
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):  # 이미지 파일만 처리
        image_path = os.path.join(image_folder, filename)  # 이미지 파일의 전체 경로
        image = cv2.imread(image_path)  # 이미지 로드

        # 해당하는 라벨 데이터 로드
        label_filename = filename.replace('.jpg', '.txt')  # 라벨 파일 이름
        label_path = os.path.join(label_folder, label_filename)  # 라벨 파일의 전체 경로

        # 파일을 라인별로 읽어들여서 처리
        try:
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
                # lines에 있는 정보를 파싱해서 필요한 데이터 추출
                # 이 부분은 txt 파일의 형식에 따라 파싱하는 과정을 작성해야 합니다.
                # txt 파일의 구조에 따라 정보 추출 및 처리를 적절하게 수정하세요.

                # 예시: lines에서 필요한 정보를 읽어서 리스트에 추가
                data.append((image, lines))  # 예시로 image와 lines를 튜플로 추가
        except FileNotFoundError as e:
            print(f"File not found error: {e}")  # 파일을 찾을 수 없는 경우 에러 메시지 출력
        except Exception as ex:
            print(f"Error occurred: {ex}")  # 기타 예외 발생 시 에러 메시지 출력
            # 추가로 필요한 예외 처리 코드를 작성할 수 있습니다.

# 학습 데이터를 파일로 저장
with open('C:\\Users\\poip8\\Desktop\\cherry_tomato.v2i.yolov5pytorch\\train\\training_data.pkl', 'wb') as f:
    pickle.dump(data, f)
    print("끝")
