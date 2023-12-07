import pyrealsense2 as rs
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# yolov5 패키지 파일 위치 추가
sys.path.append(r'C:\Users\ajw1\Desktop\yolov5')

# YOLOv5 모델 로드
model_path = Path(r'C:\Users\ajw1\Desktop\yolov5\runs\train\exp\weights\best.pt')
model = torch.load(model_path)['model'].to(torch.float32)  # 모델은 'model' 키에 저장되어 있음

# 리얼센스 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 카메라 설정 (해상도, 포맷, 프레임레이트 등)

# 카메라 시작
profile = pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 카메라 프레임을 OpenCV 포맷으로 변환하고, 입력 데이터의 타입을 변경
        color_image = np.asanyarray(color_frame.get_data())
        color_image_tensor = torch.from_numpy(color_image).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # 모델에 프레임 전달하여 객체 감지
        with torch.no_grad():
            results = model(color_image_tensor)  # 모델에 프레임 전달하여 객체 감지

        detections = results[0]  # 모델 출력에서 객체 감지 정보 가져오기

        for det in detections:
            # 각 객체의 정보 (클래스, confidence, 좌표 등) 출력
            print(det)

        cv2.imshow('YOLOv5 Object Detection', color_image)  # 결과 출력

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

finally:
    # 카메라 정리 및 종료
    pipeline.stop()
    cv2.destroyAllWindows()
