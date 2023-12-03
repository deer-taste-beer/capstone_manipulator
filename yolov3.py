# 필요한 라이브러리 임포트
import cv2
import numpy as np

# YOLOv3 설정 파일과 가중치 파일 경로
yolo_config = "path/to/yolov3.cfg"
yolo_weights = "path/to/yolov3.weights"
yolo_classes = "path/to/yolov3.txt"

# Darknet YOLO 로드
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# 클래스 이름 로드
classes = []
with open(yolo_classes, 'r') as f:
    classes = f.read().splitlines()

# 테스트할 이미지 로드
image = cv2.imread("path/to/your/image.jpg")
height, width, _ = image.shape

# 이미지 전처리 및 객체 감지
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
outs = net.forward(output_layers)

# 객체를 감지하고 클래스별로 필터링
conf_threshold = 0.5  # Confidence(신뢰도) 임계값
class_ids = []
boxes = []
confidences = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 겹치는 박스를 제거하기 위한 Non-max suppression 적용
nms_threshold = 0.4  # Non-max suppression(중복 억제) 임계값
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# 결과 출력
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    
    # 토마토의 클래스 ID를 찾아서 익은 것과 익지 않은 것 구분
    if classes[class_ids[i]] == 'tomato':
        # 익은 토마토와 익지 않은 토마토를 구분하는 추가적인 로직을 구현하세요.
        # 여기에 해당 로직을 추가해주세요.
        
        # 예시: 익은 토마토라고 판단되면 녹색 경계 상자로 표시
        color = (0, 255, 0)  # 녹색
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 결과 이미지 보기
cv2.imshow("YOLOv3 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()