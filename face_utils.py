import os
import cv2
import numpy as np
import face_recognition

def load_face_library(face_lib_path):
    # （这里放置 load_face_library 函数的代码）
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

def process_frame(frame, face_library_encodings, face_library_images, face_library_infos, threshold=1.0, detection_model="cnn"):
    # （这里放置 process_frame 函数的代码）
    face_locations = face_recognition.face_locations(frame, model=detection_model)
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
    # （这里放置 draw_results 函数的代码）
    for res in results:
        top, right, bottom, left = res["face_location"]
        color = (0, 255, 0) if res["match_found"] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = res["matched_face_info"] if res["match_found"] else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame