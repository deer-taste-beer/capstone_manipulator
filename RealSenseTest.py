import pyrealsense2 as rs
import numpy as np
import cv2

# 카메라를 활성화합니다.
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 카메라 스트림 시작
profile = pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()

        # 깊이와 이미지 프레임 추출
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 깊이 데이터와 이미지 데이터를 numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 화면에 깊이와 이미지 데이터 표시
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('Color Image', color_image)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 카메라 정리 및 종료
    pipeline.stop()
    cv2.destroyAllWindows()
