import cv2

# 视频文件路径
video_path = '/data/wyliang/datasets/ucf101/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'

# 使用OpenCV读取视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# 打印帧率信息
print(f"视频帧率：{frame_rate}")

# 关闭视频捕捉对象
cap.release()