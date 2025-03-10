import os
import cv2
import numpy as np
import face_recognition

def load_face_library(face_lib_path):
    # 加载人脸库，返回每张图片的人脸编码、图片数据以及图片名称
    face_encodings = []
    face_images = []
    face_infos = []
    for file in os.listdir(face_lib_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(face_lib_path, file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0])
                face_images.append(image)
                face_infos.append(file)
            else:
                print(f"未检测到人脸：{file}")
    return face_encodings, face_images, face_infos

def process_frame_scaled(frame, face_library_encodings, face_library_images, face_library_infos,
                         threshold=1, detection_model="cnn", scale_factor=0.5):
    """
    优化版本：先将图像缩小检测，再将检测到的坐标映射回原图进行后续处理
    参数:
      frame: 原始图像（RGB格式的NumPy数组）
      face_library_encodings, face_library_images, face_library_infos: 人脸库数据
      threshold: 比对阈值
      detection_model: 人脸检测模型 ("cnn" 或 "hog")
      scale_factor: 缩放因子（例如0.5表示将图像缩小50%）
    """
    # 缩小图像以减少计算量
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # 在缩小后的图像上进行人脸检测
    small_face_locations = face_recognition.face_locations(small_frame, model=detection_model)
    
    # 将缩小图像中检测到的人脸坐标映射回原图坐标
    face_locations = []
    for (top, right, bottom, left) in small_face_locations:
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)
        face_locations.append((top, right, bottom, left))
    
    # 在原图上根据映射回来的坐标提取人脸编码
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    results = []
    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(face_library_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < threshold:
                match_found = True
                matched_face_image = face_library_images[best_match_index]
                matched_face_info = face_library_infos[best_match_index]
            else:
                match_found = False
                matched_face_image = None
                matched_face_info = None
        else:
            match_found = False
            matched_face_image = None
            matched_face_info = None

        results.append({
            "face_location": face_location,
            "match_found": match_found,
            "matched_face_info": matched_face_info,
            "matched_face_image": matched_face_image,
            "distance": distances[best_match_index] if len(distances) > 0 else None
        })
    return results

def draw_results(frame, results):
    # 在图像上绘制检测到的人脸区域和匹配结果标注
    for res in results:
        top, right, bottom, left = res["face_location"]
        color = (0, 255, 0) if res["match_found"] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = res["matched_face_info"] if res["match_found"] else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame