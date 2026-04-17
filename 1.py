# @time     :2020/9/14 22:35
# @author   :Zohar
# 实时摄像头皮影戏版本 - 关节对位精确修正版

import os
import cv2
import numpy as np
import mediapipe as mp
import time
from PIL import Image
import math

# MediaPipe姿态检测初始- 新版API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp_base

# 皮影素材路径映射
body_img_path_map = {
    'head': './shadow_play_material/head.jpg',
    'body': './shadow_play_material/body.jpg',
    'right_hip': './shadow_play_material/right_hip.jpg',
    'right_knee': './shadow_play_material/right_knee.jpg',
    'left_hip': './shadow_play_material/left_hip.jpg',
    'left_knee': './shadow_play_material/left_knee.jpg',
    'right_elbow': './shadow_play_material/right_elbow.jpg',
    'right_wrist': './shadow_play_material/right_wrist.jpg',
    'left_elbow': './shadow_play_material/left_elbow.jpg',
    'left_wrist': './shadow_play_material/left_wrist.jpg'
}

background_img_path = './background.jpg'

def mediapipe_pose_detection_realtime(image):
    """使用MediaPipe进行实时姿态检测并转换为标准格式"""
    # 创建姿态检测器选项
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            # 转换图像格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp_base.Image(image_format=mp_base.ImageFormat.SRGB, data=image_rgb)
            
            # 执行检测
            result = landmarker.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                height, width = image.shape[:2]
                
                # 转换为像素坐标
                def get_point(landmark_id):
                    return (
                        int(landmarks[landmark_id].x * width),
                        int(landmarks[landmark_id].y * height)
                    )
                
                # 提取关键点 (新版API索引)
                nose = get_point(0)  # NOSE
                left_shoulder = get_point(11)  # LEFT_SHOULDER
                right_shoulder = get_point(12)  # RIGHT_SHOULDER
                left_elbow = get_point(13)  # LEFT_ELBOW
                right_elbow = get_point(14)  # RIGHT_ELBOW
                left_wrist = get_point(15)  # LEFT_WRIST
                right_wrist = get_point(16)  # RIGHT_WRIST
                left_hip = get_point(23)  # LEFT_HIP
                right_hip = get_point(24)  # RIGHT_HIP
                left_knee = get_point(25)  # LEFT_KNEE
                right_knee = get_point(26)  # RIGHT_KNEE
                left_ankle = get_point(27)  # LEFT_ANKLE
                right_ankle = get_point(28)  # RIGHT_ANKLE
                
                # 计算中心点
                neck = (
                    (left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2
                )
                pelvis = (
                    (left_hip[0] + right_hip[0]) // 2,
                    (left_hip[1] + right_hip[1]) // 2
                )
                
                # 计算头顶点（鼻子上方一定距离）
                head_height = abs(neck[1] - nose[1]) * 0.8
                head_top = (nose[0], int(nose[1] - head_height))
                
                # 返回标准格式的关键点数据
                return [{
                    'data': {
                        'head_top': head_top,
                        'upper_neck': neck,
                        'nose': nose,
                        'left_shoulder': left_shoulder,
                        'right_shoulder': right_shoulder,
                        'left_elbow': left_elbow,
                        'right_elbow': right_elbow,
                        'left_wrist': left_wrist,
                        'right_wrist': right_wrist,
                        'pelvis': pelvis,
                        'left_hip': left_hip,
                        'right_hip': right_hip,
                        'left_knee': left_knee,
                        'right_knee': right_knee,
                        'left_ankle': left_ankle,
                        'right_ankle': right_ankle
                    }
                }]
    except Exception as e:
        print(f"姿态检测错误: {e}")
    
    return None

def get_true_angel(value):
    return value/np.pi*180

def get_angle(x1, y1, x2, y2):
    '''计算旋转角度 - 简化版本'''
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle if angle >= 0 else angle + 360

def rotate_bound(image, angle, key_point_y):
    '''旋转图像，并取得关节点偏移量'''
    (h, w) = image.shape[:2]
    (cx, cy) = (w/2, h/2)
    (kx, ky) = cx, key_point_y
    d = abs(ky - cy)
    
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
    
    move_x = nW/2 + np.sin(angle/180*np.pi)*d 
    move_y = nH/2 - np.cos(angle/180*np.pi)*d
    
    M[0, 2] += (nW/2) - cx
    M[1, 2] += (nH/2) - cy

    return cv2.warpAffine(image, M, (nW, nH)), int(move_x), int(move_y)

def get_distences(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def append_img_by_sk_points(img, append_img_path, key_point_y, first_point, second_point, 
                           append_img_reset_width=None, append_img_max_height_rate=1, 
                           middle_flip=False, append_img_max_height=None):
    '''将需要添加的肢体图片进行缩放和定位'''
    try:
        append_image = cv2.imdecode(np.fromfile(append_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if append_image is None:
            return img
            
        # 根据长度进行缩放
        sk_height = int(get_distences(first_point[0], first_point[1], second_point[0], second_point[1]) * append_img_max_height_rate)
        
        # 缩放制约
        if append_img_max_height:
            sk_height = min(sk_height, append_img_max_height)
        
        sk_height = max(1, sk_height)  # 确保高度至少为1
        
        if append_img_reset_width is None:
            sk_width = max(1, int(sk_height / append_image.shape[0] * append_image.shape[1]))
        else:
            sk_width = max(1, int(append_img_reset_width))
        
        # 关键点映射 - 修正计算
        if append_image.shape[0] > 0:
            key_point_y_new = int(key_point_y / append_image.shape[0] * sk_height)
        else:
            key_point_y_new = key_point_y
            
        # 缩放图片
        append_image = cv2.resize(append_image, (sk_width, sk_height))
        
        img_height, img_width = img.shape[:2]
        
        # 中间翻转处理
        if middle_flip:
            middle_x = int(img_width / 2)
            if first_point[0] < middle_x and second_point[0] < middle_x:
                append_image = cv2.flip(append_image, 1)
        
        # 计算旋转角度
        angle = get_angle(first_point[0], first_point[1], second_point[0], second_point[1])
        append_image, move_x, move_y = rotate_bound(append_image, angle=angle, key_point_y=key_point_y_new)
        
        app_img_height, app_img_width = append_image.shape[:2]
        
        # 计算放置位置
        zero_x = first_point[0] - move_x
        zero_y = first_point[1] - move_y
        
        # 确保append_image有3个通道
        if len(append_image.shape) == 3:
            (b, g, r) = cv2.split(append_image)
            
            # 优化的像素复制
            for i in range(app_img_height):
                for j in range(app_img_width):
                    target_y = zero_y + i
                    target_x = zero_x + j
                    
                    # 边界检查
                    if (0 <= target_y < img_height and 0 <= target_x < img_width and
                        200 < r[i][j] < 230):  # 调整颜色阈值
                        img[target_y][target_x] = append_image[i][j]
        
        return img
        
    except Exception as e:
        print(f"处理图片 {append_img_path} 时出错: {e}")
        return img

def get_combine_img_realtime(image, result, body_img_path_map, backgroup_img_path='background.jpg'):
    '''实时识别图片中的关节点，并将皮影的肢体进行对应'''
    if result is None or len(result) == 0:
        return None
        
    try:
        # 背景图片
        backgroup_image = cv2.imread(backgroup_img_path)
        if backgroup_image is None:
            backgroup_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 200
        
        image_flag = cv2.resize(backgroup_image, (image.shape[1], image.shape[0]))
        
        data = result[0]['data']
        
        # 计算基础尺寸
        head_neck_dist = get_distences(data['head_top'][0], data['head_top'][1],
                                     data['upper_neck'][0], data['upper_neck'][1])
        min_width = max(10, int(head_neck_dist / 3))
        
        # 肩膀宽度
        shoulder_width = get_distences(data['left_shoulder'][0], data['left_shoulder'][1],
                                     data['right_shoulder'][0], data['right_shoulder'][1])
        
        # 右大腿
        append_img_reset_width = max(int(shoulder_width * 0.8), min_width)
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['right_hip'], 
                                           key_point_y=10, first_point=data['right_hip'],
                                           second_point=data['right_knee'], 
                                           append_img_reset_width=append_img_reset_width)
        
        # 右小腿
        append_img_reset_width = max(int(shoulder_width * 0.6), min_width)
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['right_knee'], 
                                           key_point_y=10, first_point=data['right_knee'],
                                           second_point=data['right_ankle'], 
                                           append_img_reset_width=append_img_reset_width)
        
        # 左大腿
        append_img_reset_width = max(int(shoulder_width * 0.8), min_width)
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['left_hip'], 
                                           key_point_y=10, first_point=data['left_hip'],
                                           second_point=data['left_knee'], 
                                           append_img_reset_width=append_img_reset_width)
        
        # 左小腿
        append_img_reset_width = max(int(shoulder_width * 0.6), min_width)
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['left_knee'], 
                                           key_point_y=10, first_point=data['left_knee'],
                                           second_point=data['left_ankle'], 
                                           append_img_reset_width=append_img_reset_width)
        
        # 右上臂
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['right_elbow'], 
                                           key_point_y=25, first_point=data['right_shoulder'],
                                           second_point=data['right_elbow'], 
                                           append_img_max_height_rate=1.0)
        
        # 右前臂
        upper_arm_length = get_distences(data['right_shoulder'][0], data['right_shoulder'][1],
                                        data['right_elbow'][0], data['right_elbow'][1])
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['right_wrist'], 
                                           key_point_y=10, first_point=data['right_elbow'],
                                           second_point=data['right_wrist'], 
                                           append_img_max_height_rate=1.0,
                                           append_img_max_height=int(upper_arm_length * 1.2))
        
        # 左上臂
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['left_elbow'], 
                                           key_point_y=25, first_point=data['left_shoulder'],
                                           second_point=data['left_elbow'], 
                                           append_img_max_height_rate=1.0)
        
        # 左前臂
        upper_arm_length = get_distences(data['left_shoulder'][0], data['left_shoulder'][1],
                                        data['left_elbow'][0], data['left_elbow'][1])
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['left_wrist'], 
                                           key_point_y=10, first_point=data['left_elbow'],
                                           second_point=data['left_wrist'], 
                                           append_img_max_height_rate=1.0,
                                           append_img_max_height=int(upper_arm_length * 1.2))
        
        # 头部
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['head'], 
                                           key_point_y=10, first_point=data['head_top'],
                                           second_point=data['upper_neck'], 
                                           append_img_max_height_rate=1.0, middle_flip=True)
        
        # 身体
        append_img_reset_width = max(int(shoulder_width * 1.2), min_width * 3)
        image_flag = append_img_by_sk_points(image_flag, body_img_path_map['body'], 
                                           key_point_y=20, first_point=data['upper_neck'],
                                           second_point=data['pelvis'], 
                                           append_img_reset_width=append_img_reset_width, 
                                           append_img_max_height_rate=1.0)
        
        return image_flag
        
    except Exception as e:
        print(f"合成皮影图像时出错: {e}")
        return None

def realtime_shadow_play():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("实时皮影戏开始！按 'q' 键退出")
    print("左侧窗口：摄像头画面，右侧窗口：纯皮影戏效果")
    
    # 创建两个窗口并设置位置
    cv2.namedWindow('摄像头画面', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('皮影戏效果', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('摄像头画面', 100, 100)
    cv2.moveWindow('皮影戏效果', 750, 100)
    
    # 创建默认皮影背景
    default_shadow = cv2.imread(background_img_path)
    if default_shadow is not None:
        default_shadow = cv2.resize(default_shadow, (640, 480))
    else:
        default_shadow = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # 性能优化设置
    frame_count = 0
    process_every_n_frames = 2  # 每2帧处理一次
    last_shadow_result = default_shadow.copy()
    
    # FPS计算
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 水平翻转画面（镜像效果）
        frame = cv2.flip(frame, 1)
        
        # FPS计算
        fps_frame_count += 1
        if fps_frame_count % 30 == 0:
            fps_end_time = time.time()
            fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            print(f"FPS: {fps:.1f}")
        
        # 显示摄像头画面
        cv2.imshow('摄像头画面', frame)
        
        # 姿态检测和皮影生成
        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            result = mediapipe_pose_detection_realtime(frame)
            
            if result is not None:
                try:
                    combined_img = get_combine_img_realtime(frame, result, body_img_path_map, background_img_path)
                    if combined_img is not None:
                        last_shadow_result = cv2.resize(combined_img, (640, 480))
                except Exception as e:
                    print(f"处理错误: {e}")
        
        # 显示皮影效果
        cv2.imshow('皮影戏效果', last_shadow_result)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("实时皮影戏结束")

if __name__ == "__main__":
    realtime_shadow_play()