import os
# 用于访问操作系统功能，比如遍历目录、拼接文件路径等
import cv2
# 导入opencv库，用于图像处理、绘制图形、显示图片等
import numpy as np
import face_recognition
# 该库封装了基于dlib的人脸检测和人脸编码提取方法

# 这个函数用来遍历指定的人脸目录，加载图片并计算人脸编码（128维向量）
def load_face_library(face_lib_path):
    # （这里放置 load_face_library 函数的代码）
    face_encodings = []
    # 存放每张图片中检测到的人脸编码
    face_images = []
    # 存放加载的图片数据（数组形式）
    face_infos = []
    # 存放附加信息（例如图片名称），方便后续使用
    for file in os.listdir(face_lib_path):
    # 使用os.listdir遍历目录下所有文件，过滤出后缀为.jpg、.jpeg和.png的文件
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(face_lib_path, file)
            # 使用os.path.join将目录路径与文件名拼接成完整的图片路径
            image = face_recognition.load_image_file(img_path)
            # 使用face_recognition.load_image_file加载图片数据
            encodings = face_recognition.face_encodings(image)
            # 使用face_recognition.face_encodings计算图片中的人脸编码
            if encodings:
            # 如果encodings不为空，说明图片中检测到了至少一张人脸
                face_encodings.append(encodings[0])
                # 将检测到的人脸编码添加到face_encodings列表中
                face_images.append(image)
                # 将加载的图片数据添加到face_images列表中
                face_infos.append(file)
                # 将图片名称添加到face_infos列表中
            else:
                print(f"未检测到人脸：{file}")
    return face_encodings, face_images, face_infos

# 用于对一张输入的图片（视频帧）进行人脸检测、编码提取，并与人脸库中的编码进行比对
# frame 输入图片（通常为Numpy数组格式）
# face_library_encodings 人脸库中的编码列表
# face_library_images 人脸库中的图片列表
# face_library_infos 人脸库中的附加信息列表
# threshold 设定的比对阈值，距离小于该值时认为匹配成功（默认0.6）
def process_frame(frame, face_library_encodings, face_library_images, face_library_infos, threshold=0.6):
    # （这里放置 process_frame 函数的代码）
    face_locations = face_recognition.face_locations(frame)
    # 检测图片中所有人脸的位置，返回的每个位置时一个四元组（top，right，bottom，left），表示人脸在图片中的矩形边界
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # 根据检测到的人脸位置，提取每个脸部的128维编码，便于后续进行比对
    results = []
    # 创建一个空列表results用于存储每张人脸的比对结果
    for (face_location, face_encoding) in zip(face_locations, face_encodings):
    # 使用zip将每个检测到的人脸的位置和对应的编码组合在一起，逐个进行处理
        distances = face_recognition.face_distance(face_library_encodings, face_encoding)
        # 计算输入人脸编码与人脸库中所有编码之间的欧式距离
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            # 找到距离最小的索引，即最匹配的候选
            if distances[best_match_index] < threshold:
                match_found = True
                matched_face_image = face_library_images[best_match_index]
                matched_face_info = face_library_infos[best_match_index]
            else:
                match_found = False
                matched_face_image = None
                matched_face_info = None
            # 这一段代码是判断是否匹配成功
        else:
            match_found = False
            matched_face_image = None
            matched_face_info = None
        # 这段是异常处理，如果没有计算出距离（例如人脸库为空），则直接标记匹配失败

        results.append({
            "face_location": face_location,
            "match_found": match_found,
            "matched_face_info": matched_face_info,
            "matched_face_image": matched_face_image,
            "distance": distances[best_match_index] if len(distances) > 0 else None
        })
        # 这段代码是将当前人脸的位置信息、匹配结果、匹配到的库中的人脸信息、对应图片以及最小距离，存入字典，并添加到results列表中
    return results
    # 函数返回一个包含所有处理人脸结果的列表

# 用于在输入的图片（视频帧）上绘制人脸检测结果和匹配结果
def draw_results(frame, results):
    # （这里放置 draw_results 函数的代码）
    for res in results:
        top, right, bottom, left = res["face_location"]
        # 对每个检测结果，先从字典中提取人脸的边界坐标
        color = (0, 255, 0) if res["match_found"] else (0, 0, 255)
        # 绿色为匹配成功，红色为匹配失败
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # 在图像上绘制矩形框，标记出人脸区域，参数中的2表示线条宽度
        label = res["matched_face_info"] if res["match_found"] else "Unknown"
        # 匹配成功则显示对应人脸信息（例如文件名），否则显示“Unknown”
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # 将标签绘制在矩形框上方（left，top-10）位置，字体、大小和颜色与矩形框保持一致
    return frame
    # 函数返回已绘制上检测和匹配信息的图像