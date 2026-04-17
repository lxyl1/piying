#!/usr/bin/python
# @time     :2020/9/14 22:35
# @author   :Zohar
# 实时摄像头皮影戏版本 - 性能优化版

import os
import cv2
import numpy as np
import mediapipe as mp
import time

# MediaPipe姿态检测初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

body_img_path_map = {
    "right_hip" : "./shadow_play_material/right_hip.jpg",
    "right_knee" : "./shadow_play_material/right_knee.jpg",
    "left_hip" : "./shadow_play_material/left_hip.jpg",
    "left_knee" : "./shadow_play_material/left_knee.jpg",
    "left_elbow" : "./shadow_play_material/left_elbow.jpg",
    "left_wrist" : "./shadow_play_material/left_wrist.jpg",
    "right_elbow" : "./shadow_play_material/right_elbow.jpg",
    "right_wrist" : "./shadow_play_material/right_wrist.jpg",
    "head" : "./shadow_play_material/head.jpg",
    "body" : "./shadow_play_material/body.jpg"
}
background_img_path = './background.jpg'

# 预加载所有图片资源
body_images_cache = {}
for key, path in body_img_path_map.items():
    try:
        body_images_cache[key] = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        print(f"警告：无法加载图片 {path}")
        body_images_cache[key] = None

# 预加载背景图片
try:
    background_image_cache = cv2.imread(background_img_path)
except:
    print(f"警告：无法加载背景图片 {background_img_path}")
    background_image_cache = None

# MediaPipe关键点映射到原始关键点名称
MEDIAPIPE_POSE_MAPPING = {
    'head_top': mp_pose.PoseLandmark.NOSE,
    'upper_neck': mp_pose.PoseLandmark.NOSE,
    'pelvis': mp_pose.PoseLandmark.LEFT_HIP,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST
}

def mediapipe_pose_detection_realtime(image, pose_detector):
    """
    使用MediaPipe进行实时姿态检测 - 优化版
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width = image.shape[:2]
        
        pose_data = {}
        for key, mp_landmark in MEDIAPIPE_POSE_MAPPING.items():
            landmark = landmarks[mp_landmark.value]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            pose_data[key] = [x, y]
        
        # 特殊处理：计算骨盆中心点
        left_hip = pose_data['left_hip']
        right_hip = pose_data['right_hip']
        pose_data['pelvis'] = [
            int((left_hip[0] + right_hip[0]) / 2),
            int((left_hip[1] + right_hip[1]) / 2)
        ]
        
        # 特殊处理：使用肩膀中点作为上颈部
        left_shoulder = pose_data['left_shoulder']
        right_shoulder = pose_data['right_shoulder']
        pose_data['upper_neck'] = [
            int((left_shoulder[0] + right_shoulder[0]) / 2),
            int((left_shoulder[1] + right_shoulder[1]) / 2)
        ]
        
        return [{'data': pose_data}]
    else:
        return None

def get_true_angel(value):
    return value/np.pi*180

def get_angle(x1, y1, x2, y2):
    dx = abs(x1- x2)
    dy = abs(y1- y2)
    result_angele = 0
    if x1 == x2:
        if y1 > y2:
            result_angele = 180
    else:
        if y1!=y2:
            the_angle = int(get_true_angel(np.arctan(dx/dy)))
        if x1 < x2:
            if y1>y2:
                result_angele = -(180 - the_angle)
            elif y1<y2:
                result_angele = -the_angle
            elif y1==y2:
                result_angele = -90
        elif x1 > x2:
            if y1>y2:
                result_angele = 180 - the_angle
            elif y1<y2:
                result_angele = the_angle
            elif y1==y2:
                result_angele = 90
    
    if result_angele<0:
        result_angele = 360 + result_angele
    return result_angele

def rotate_bound(image, angle, key_point_y):
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    (kx,ky) = cx, key_point_y
    d = abs(ky - cy)
    
    M = cv2.getRotationMatrix2D((cx,cy), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    move_x = nW/2 + np.sin(angle/180*np.pi)*d 
    move_y = nH/2 - np.cos(angle/180*np.pi)*d
    
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy

    return cv2.warpAffine(image,M,(nW,nH)), int(move_x), int(move_y)

def get_distences(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def append_img_by_sk_points_optimized(img, body_part_key, key_point_y, first_point, second_point, 
                                     append_img_reset_width=None, append_img_max_height_rate=1, 
                                     middle_flip=False, append_img_max_height=None):
    """
    优化版的图片合成函数
    """
    # 使用缓存的图片
    append_image = body_images_cache.get(body_part_key)
    if append_image is None:
        return img
    
    append_image = append_image.copy()  # 复制以避免修改原始缓存

    # 根据长度进行缩放
    sk_height = int(get_distences(first_point[0], first_point[1], second_point[0], second_point[1])*append_img_max_height_rate)
    if append_img_max_height:
        sk_height = min(sk_height, append_img_max_height)

    sk_width = int(sk_height/append_image.shape[0]*append_image.shape[1]) if append_img_reset_width is None else int(append_img_reset_width)
    if sk_width <= 0:
        sk_width = 1
    if sk_height <= 0:
        sk_height = 1

    # 关键点映射
    key_point_y_new = int(key_point_y/append_image.shape[0]*append_image.shape[1])
    # 缩放图片
    append_image = cv2.resize(append_image, (sk_width, sk_height))

    img_height, img_width, _ = img.shape
    # 翻转处理
    if middle_flip:
        middle_x = int(img_width/2)
        if first_point[0] < middle_x and second_point[0] < middle_x:
            append_image = cv2.flip(append_image, 1)

    # 旋转角度
    angle = get_angle(first_point[0], first_point[1], second_point[0], second_point[1])
    append_image, move_x, move_y = rotate_bound(append_image, angle=angle, key_point_y=key_point_y_new)
    
    zero_x = first_point[0] - move_x
    zero_y = first_point[1] - move_y

    # 优化的图片合成 - 使用向量化操作替代双重循环
    app_h, app_w = append_image.shape[:2]
    
    # 计算有效区域
    start_y = max(0, zero_y)
    end_y = min(img_height, zero_y + app_h)
    start_x = max(0, zero_x)
    end_x = min(img_width, zero_x + app_w)
    
    if start_y < end_y and start_x < end_x:
        # 计算在append_image中的对应区域
        app_start_y = start_y - zero_y
        app_end_y = end_y - zero_y
        app_start_x = start_x - zero_x
        app_end_x = end_x - zero_x
        
        # 提取区域
        region = append_image[app_start_y:app_end_y, app_start_x:app_end_x]
        if len(region.shape) == 3 and region.shape[2] == 3:
            # 创建掩码 - 只处理特定颜色范围的像素
            mask = (region[:,:,2] > 200) & (region[:,:,2] < 230)  # R通道
            
            # 使用掩码进行批量赋值
            img[start_y:end_y, start_x:end_x][mask] = region[mask]
    
    return img

def get_combine_img_realtime_optimized(image, result):
    """
    优化版的实时图片合成函数
    """
    if result is None:
        return None
        
    # 使用缓存的背景图片
    if background_image_cache is not None:
        image_flag = cv2.resize(background_image_cache, (image.shape[1], image.shape[0]))
    else:
        image_flag = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 最小宽度
    min_width = int(get_distences(result[0]['data']['head_top'][0], result[0]['data']['head_top'][1],
                result[0]['data']['upper_neck'][0], result[0]['data']['upper_neck'][1])/3)

    # 使用优化后的函数
    #右大腿
    append_img_reset_width = max(int(get_distences(result[0]['data']['pelvis'][0], result[0]['data']['pelvis'][1],
                                            result[0]['data']['left_hip'][0], result[0]['data']['right_hip'][1])*1.6), min_width)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'right_hip', key_point_y=10, first_point=result[0]['data']['right_hip'],
                                        second_point=result[0]['data']['right_knee'], append_img_reset_width=append_img_reset_width)

    # 右小腿
    append_img_reset_width = max(int(get_distences(result[0]['data']['pelvis'][0], result[0]['data']['pelvis'][1],
                                            result[0]['data']['left_hip'][0], result[0]['data']['right_hip'][1])*1.5), min_width)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'right_knee', key_point_y=10, first_point=result[0]['data']['right_knee'],
                                            second_point=result[0]['data']['right_ankle'], append_img_reset_width=append_img_reset_width)

    # 左大腿
    append_img_reset_width = max(int(get_distences(result[0]['data']['pelvis'][0], result[0]['data']['pelvis'][1],
                                            result[0]['data']['left_hip'][0], result[0]['data']['left_hip'][1])*1.6), min_width)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'left_hip', key_point_y=0, first_point=result[0]['data']['left_hip'],
                                        second_point=result[0]['data']['left_knee'], append_img_reset_width=append_img_reset_width)

    # 左小腿
    append_img_reset_width = max(int(get_distences(result[0]['data']['pelvis'][0], result[0]['data']['pelvis'][1],
                                            result[0]['data']['left_hip'][0], result[0]['data']['left_hip'][1])*1.5), min_width)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'left_knee', key_point_y=10, first_point=result[0]['data']['left_knee'],
                                            second_point=result[0]['data']['left_ankle'], append_img_reset_width=append_img_reset_width)

    # 右手臂
    image_flag = append_img_by_sk_points_optimized(image_flag, 'left_elbow', key_point_y=25, first_point=result[0]['data']['right_shoulder'],
                                        second_point=result[0]['data']['right_elbow'], append_img_max_height_rate=1.2)

    # 右手肘
    append_img_max_height = int(get_distences(result[0]['data']['right_shoulder'][0], result[0]['data']['right_shoulder'][1],
                                            result[0]['data']['right_elbow'][0], result[0]['data']['right_elbow'][1])*1.6)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'left_wrist', key_point_y=10, first_point=result[0]['data']['right_elbow'],
                                            second_point=result[0]['data']['right_wrist'], append_img_max_height_rate=1.5, 
                                            append_img_max_height=append_img_max_height)

    # 左手臂
    image_flag = append_img_by_sk_points_optimized(image_flag, 'right_elbow', key_point_y=25, first_point=result[0]['data']['left_shoulder'], 
                                        second_point=result[0]['data']['left_elbow'],  append_img_max_height_rate=1.2)

    # 左手肘
    append_img_max_height = int(get_distences(result[0]['data']['left_shoulder'][0], result[0]['data']['left_shoulder'][1],
                                        result[0]['data']['left_elbow'][0], result[0]['data']['left_elbow'][1])*1.6)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'right_wrist', key_point_y=10, first_point=result[0]['data']['left_elbow'],
                                        second_point=result[0]['data']['left_wrist'], append_img_max_height_rate=1.5, 
                                         append_img_max_height=append_img_max_height)

    # 头
    image_flag = append_img_by_sk_points_optimized(image_flag, 'head', key_point_y=10, first_point=result[0]['data']['head_top'],
                    second_point=result[0]['data']['upper_neck'], append_img_max_height_rate=1.2, middle_flip=True)

    # 身体
    append_img_reset_width = max(int(get_distences(result[0]['data']['left_shoulder'][0], result[0]['data']['left_shoulder'][1],
                                            result[0]['data']['right_shoulder'][0], result[0]['data']['right_shoulder'][1])*1.2), min_width*3)
    image_flag = append_img_by_sk_points_optimized(image_flag, 'body', key_point_y=20, first_point=result[0]['data']['upper_neck'],
                    second_point=result[0]['data']['pelvis'], append_img_reset_width=append_img_reset_width, append_img_max_height_rate=1.2)
    
    return image_flag

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
    
    # 初始化MediaPipe - 只创建一次
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 使用默认复杂度(1)避免lite模型下载失败
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 创建默认皮影背景
    if background_image_cache is not None:
        default_shadow = cv2.resize(background_image_cache, (640, 480))
    else:
        default_shadow = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 性能优化：降低处理频率
    frame_count = 0
    process_every_n_frames = 3  # 每3帧处理一次姿态检测（进一步降低频率）
    last_shadow_result = default_shadow.copy()
    
    # 添加FPS计算
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 水平翻转画面（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 显示摄像头画面
            cv2.imshow('摄像头画面', frame)
            
            # 性能优化：不是每帧都进行姿态检测
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                # 进行姿态检测
                result = mediapipe_pose_detection_realtime(frame, pose_detector)
                
                if result is not None:
                    # 生成皮影戏效果
                    try:
                        combined_img = get_combine_img_realtime_optimized(frame, result)
                        if combined_img is not None:
                            last_shadow_result = cv2.resize(combined_img, (640, 480))
                    except Exception as e:
                        print(f"处理错误: {e}")
            
            # 显示皮影效果
            cv2.imshow('皮影戏效果', last_shadow_result)
            
            # FPS计算
            fps_counter += 1
            if fps_counter % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_start_time)
                print(f"当前FPS: {fps:.1f}")
                fps_start_time = current_time
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放资源
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("实时皮影戏结束")

if __name__ == "__main__":
    realtime_shadow_play()