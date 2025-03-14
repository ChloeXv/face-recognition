import os
import cv2
import face_recognition
from face_utils import load_face_library, process_frame_scaled, draw_results

if __name__ == "__main__":
    # 加载人脸库
    face_lib_path = r"C:\Users\chkxw\OneDrive\Desktop\face_recognition\face_library"
    face_library_encodings, face_library_images, face_library_infos = load_face_library(face_lib_path)

    # 视频帧目录
    video_capture_path = r"C:\Users\chkxw\OneDrive\Desktop\face_recognition\video_capture"

    for file in sorted(os.listdir(video_capture_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame_path = os.path.join(video_capture_path, file)
            frame = face_recognition.load_image_file(frame_path)
            # 将 RGB 转换为 BGR 以供 OpenCV 使用
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 使用降低分辨率的方式进行检测
            results = process_frame_scaled(frame, face_library_encodings, face_library_images,
                                           face_library_infos, threshold=0.6, detection_model="cnn", scale_factor=0.5)
            
            print(f"Processing {file}:")
            for idx, res in enumerate(results):
                print(f"  检测到人脸 {idx+1}:")
                print("    人脸位置:", res["face_location"])
                print("    匹配结果:", "匹配成功" if res["match_found"] else "匹配失败")
                if res["match_found"]:
                    print("    匹配信息:", res["matched_face_info"])
                    print("    匹配距离:", res["distance"])
            print("---------------------------")

            result_image = draw_results(frame_bgr, results)
            cv2.imshow("Face Recognition", result_image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
