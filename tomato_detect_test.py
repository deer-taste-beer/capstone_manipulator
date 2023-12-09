import cv2
import pyrealsense2 as rs
import numpy as np
import torch

# YOLOv5 모델 로드
model = torch.hub.load(r'C:\Users\poip8\Desktop\yolov5', 'custom', path='C:/Users/poip8/Desktop/yolov5/runs/train/exp7/weights/best.pt', source='local')

# Realsense 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    # 프레임 얻기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Realsense 프레임을 OpenCV 포맷으로 변환
    frame = np.asanyarray(color_frame.get_data())

    # YOLOv5를 위한 전처리
    results = model(frame)

    # 감지된 객체에 박스와 클래스 이름 표시
    for det in results.pred[0]:
        class_name = model.names[int(det[-1])]
        bbox = det[:4].cpu().numpy().astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 영상 표시
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # 종료
    if key == ord("q"):
        break

# 리소스 해제
cv2.destroyAllWindows()
pipeline.stop()
