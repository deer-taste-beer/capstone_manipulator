import json
import shutil
import os

def copy_images_based_on_json(json_folder, image_folder, target_folder):
    # JSON 폴더 내의 모든 JSON 파일을 검색
    for json_file_name in os.listdir(json_folder):
        if json_file_name.endswith(".jpg"):
            # JSON 파일의 이름에서 확장자를 제외한 부분 추출
            image_name = os.path.splitext(json_file_name)[0]

            # 이미지 파일의 경로 구성
            image_path = os.path.join(image_folder, image_name + '.txt')

            # 이미지를 복사할 폴더로 복사
            target_path = os.path.join(target_folder, image_name + '.txt')
            try:
                shutil.copyfile(image_path, target_path)
                print(f"Image copied: {image_name}")
            except FileNotFoundError as e:
                print(f"File not found error: {e}")  # 파일을 찾을 수 없는 경우 에러 메시지 출력
            except Exception as ex:
                print(f"Error occurred: {ex}")  # 기타 예외 발생 시 에러 메시지 출력
                # 추가로 필요한 예외 처리 코드를 작성할 수 있습니다. 

# 사용 예시
json_folder_path = r'C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\labels'
image_folder_path = r'C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\따로 실험용12_06'
target_folder_path = r'C:\Users\poip8\Desktop\cherry_tomato.v2i.yolov5pytorch\train\target'

copy_images_based_on_json(image_folder_path, json_folder_path, target_folder_path)