import pyrealsense2 as rs
import torch
import cv2
import numpy as np
from pathlib import Path

# yolov5 모델 로드
# model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=Path(r"C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\training_data.pkl"))  # pkl 파일 경로로 모델 로드
model_path = Path(r"C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\training_data.pkl")
model = torch.load(model_path)

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

        # 카메라 프레임을 OpenCV 포맷으로 변환
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)  # 모델에 프레임 전달하여 객체 감지
        detections = results.pandas().xyxy[0]  # 감지된 객체 정보 가져오기

        for _, det in detections.iterrows():
            # 각 객체의 정보 (클래스, confidence, 좌표 등) 출력
            print(det)

        cv2.imshow('YOLOv5 Object Detection', color_image)  # 결과 출력

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

finally:
    # 카메라 정리 및 종료
    pipeline.stop()
    cv2.destroyAllWindows()
