import os
import cv2
import face_recognition
from face_utils_practice import load_face_library, process_frame, draw_results

if __name__ == "__main__":
    # 指定人脸库目录（Windows路径需要用原始字符串或者双反斜杠）
    face_lib_path = r"C:\Users\chkxw\OneDrive\Desktop\face_recognition\face_library"
    # 调用load_face_library函数加载人脸库，获得编码、图片和信息
    face_library_encodings, face_library_images, face_library_infos = load_face_library(face_lib_path)

    # 指定包含视频截帧的目录路径
    video_capture_path = r"C:\Users\chkxw\OneDrive\Desktop\face_recognition\video_capture"

    # 遍历视频帧目录中的每一种图片进行测试
    for file in sorted(os.listdir(video_capture_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame_path = os.path.join(video_capture_path, file)
            frame = face_recognition.load_image_file(frame_path)
            # 按照文件名顺序，对视频帧目录中的每一张图片文件进行测试，过滤出图片格式文件、拼接图片完整路径
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 转换为opencv格式（BGR）
            results = process_frame(frame, face_library_encodings, face_library_images, face_library_infos, threshold=0.6)
            # 处理当前帧，调用process_frame函数进行人脸检测和匹配
            print(f"Processing {file}:")
            for idx, res in enumerate(results):
                print(f"  检测到人脸 {idx+1}:")
                print("    人脸位置:", res["face_location"])
                print("    匹配结果:", "匹配成功" if res["match_found"] else "匹配失败")
                if res["match_found"]:
                    print("    匹配信息:", res["matched_face_info"])
                    print("    匹配距离:", res["distance"])
            print("---------------------------")
            # 打印检测信息

            result_image = draw_results(frame_bgr, results)
            cv2.imshow("Face Recognition", result_image)
            # 显示处理结果
            cv2.waitKey(0)
            # 等待用户按下任意键继续

    cv2.destroyAllWindows()
    # 关闭所有的窗口