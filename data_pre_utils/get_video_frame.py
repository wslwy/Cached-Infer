import os
import cv2

# 402
# # UCF101数据集视频文件夹路径
# video_folder = '/data/wyliang/datasets/ucf101/UCF-101'

# # 输出帧的文件夹路径
# output_frames_folder = '/data/wyliang/datasets/ucf101/UCF-101-frames'

# 407
# UCF101数据集视频文件夹路径
video_folder = '/data0/wyliang/datasets/ucf101/UCF-101'

# 输出帧的文件夹路径
output_frames_folder = '/data0/wyliang/datasets/ucf101/UCF-101-frames'

# 遍历UCF101主文件夹下的所有子文件夹
for idx, video_sub_dir in enumerate(os.listdir(video_folder)):
    sub_folder_path = os.path.join(video_folder, video_sub_dir)
    
    # 检查子文件夹是否是目录
    if os.path.isdir(sub_folder_path):
        # 在输出文件夹中创建对应的子文件夹
        output_sub_folder = os.path.join(output_frames_folder, video_sub_dir)
        os.makedirs(output_sub_folder, exist_ok=True)
        
        # 遍历子文件夹中的视频文件
        for sub_idx, video_file in enumerate(os.listdir(sub_folder_path)):
            video_path = os.path.join(sub_folder_path, video_file)
            # 在输出子文件夹中每个视频创建对应的子文件夹
            output_sub_sub_folder = os.path.join(output_sub_folder, video_file.split(".")[0])
            os.makedirs(output_sub_sub_folder, exist_ok=True)
            # 检查输出子文件夹
            # print(output_sub_sub_folder)

            
            # 使用OpenCV打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            # # 获取视频的帧率（UCF101为25Hz）
            # fps = int(cap.get(cv2.CAP_PROP_FPS))
            # if fps != 25:
            #     print("something wrong")
            
            # 遍历视频的每一帧，并保存为图像文件
            frame_count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    print("error, failed to get video frames")
                
                # 输出帧的保存路径，根据视频文件名和帧数生成对应的图像文件名
                output_frame_path = os.path.join(output_sub_sub_folder, f'{os.path.splitext(video_file)[0]}_frame_{frame_count}.jpg')
                cv2.imwrite(output_frame_path, frame)
                
                frame_count += 1
            
            cap.release()

            print(f"total:{len(os.listdir(sub_folder_path))}, {sub_idx} finished")

    print(f"{idx}, {((idx+1)/101 * 100):.2}% finished, dir: {video_sub_dir} finished")
    print("\n#################################################")

print('all fream extracted')