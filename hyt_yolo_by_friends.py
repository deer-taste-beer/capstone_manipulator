import os
import cv2
import json
import numpy as np
import pickle

# 라벨과 이미지가 저장된 폴더 여기엔 폴더 위치로 바꿔주기
label_folder = 'C:/data/labels/' 
image_folder = 'C:/data/images/'

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
        label_filename = filename.replace('.jpg', '.json')  # 라벨 파일 이름
        label_path = os.path.join(label_folder, label_filename)  # 라벨 파일의 전체 경로
        with open(label_path, 'r') as f:
            labels = json.load(f)

        # YOLO 설정 v5 버전
        net = cv2.dnn.readNet("yolov5.weights", "yolov5.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # 이미지 전처리
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # YOLO를 통한 객체 탐지
        outs = net.forward(output_layers)

        # 탐지된 객체에 대한 정보 처리
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: #이 부분도 나중에 0.5말고 판단하에 변경해야함
                    # 객체 탐지가 방울 토마토인지 확인
                    if labels[class_id] == 'tomato':
                        # 토마토의 색상을 기반으로 익었는지 판단
                        is_ripe = determine_ripeness(image)
                        print(f"Tomato detected in {filename}! Ripe: {is_ripe}")
                        # 학습 데이터에 추가
                        data.append((image, labels, is_ripe))

# 학습 데이터를 파일로 저장
with open('training_data.pkl', 'wb') as f:
    pickle.dump(data, f)


#12-4