import psycopg2
import cv2
import pandas as pd
import numpy as np
import face_recognition
import os

# 数据库连接信息
DB_PARAMS = {
    "dbname": "media-analysis_dev",
    "user": "postgres",
    "password": "a3d20$3Dss",
    "host": "172.16.52.103",
    "port": "5432"
}

# 阈值（用于人脸匹配）
MATCH_THRESHOLD = 0.6

def get_person_data():
    """ 从数据库获取人物 ID 和人脸图像路径 """
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    cursor.execute("SELECT id, person_image_path FROM public.person_info;")
    person_data = cursor.fetchall()

    cursor.close()
    conn.close()
    return person_data  # [(id1, path1), (id2, path2), ...]

def load_face_vectors(person_data):
    """ 读取人脸图像并转换为人脸向量 """
    person_vectors = []
    person_ids = []

    for person_id, image_path in person_data:
        if not os.path.exists(image_path):
            print(f"❌ file {image_path}, not exist...")
            continue

        # generate image from image path
        image = face_recognition.load_image_file(image_path)
        # get face encodings through image

        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # list of face_encodings
            person_vectors.append(face_encodings[0])  # only the first face and there should be only one face
            # list of person_ids
            person_ids.append(person_id)
        else:
            print(f"⚠️ face not found: {image_path}")

    return person_ids, person_vectors  # [id1, id2, ...], [vec1, vec2, ...]

def recognize_faces_in_frames(picture_paths, person_vectors, person_ids):
    """ 处理所有视频帧，并进行人脸匹配 """
    matched_results = []  # 存储匹配结果的列表

    for frame_path in picture_paths:
        if not os.path.exists(frame_path):
            print(f"❌ file {frame_path}, not exist...")
            continue

        # 读取视频帧图像
        frame = face_recognition.load_image_file(frame_path)
        
        # 检测人脸并提取人脸向量
        face_locations = face_recognition.face_locations(frame)  # 识别人脸位置
        face_encodings = face_recognition.face_encodings(frame, face_locations)  # 提取特征向量
        
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(person_vectors, face_encoding)  # 计算人脸距离
            best_match_index = np.argmin(distances)  # 选择最相近的人脸索引
            
            if distances[best_match_index] < MATCH_THRESHOLD:
                matched_results.append((frame_path, person_ids[best_match_index]))  # (帧路径, 匹配人脸ID)
                print(f"✅ 匹配成功: {frame_path} (Person ID: {person_ids[best_match_index]})")

    return matched_results


# 示例 picture_paths（假设已经获取）
picture_paths = [
    "/path/to/frame1.jpg",
    "/path/to/frame2.jpg",
    "/path/to/frame3.jpg"
]

# person_data = get_person_data()  # 从数据库获取人脸库信息
# person_ids, person_vectors = load_face_vectors(person_data)  # 提取人脸向量

# 示例人脸库数据（假设已经获取）
person_ids = [1, 2, 3]  # 人脸 ID
person_vectors = [
    np.random.rand(128),  # 假设是 128 维的特征向量
    np.random.rand(128),
    np.random.rand(128)
]

# 进行人脸匹配
matched_results = recognize_faces_in_frames(picture_paths, person_vectors, person_ids)

# 转换为 DataFrame 并显示
df = pd.DataFrame(matched_results, columns=["视频帧路径", "匹配的人脸ID"])
import ace_tools as tools
tools.display_dataframe_to_user(name="匹配结果", dataframe=df)
