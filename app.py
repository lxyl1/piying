import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp_base
import time
import math
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="实时皮影戏", layout="wide")

# 全局变量
_landmarker = None

def init_pose_landmarker():
    """初始化姿态检测器"""
    global _landmarker
    if _landmarker is None:
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            _landmarker = vision.PoseLandmarker.create_from_options(options)
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    return True

class ShadowPuppetTransformer(VideoTransformerBase):
    def __init__(self):
        self.landmarker = None
        self.init_detector()
    
    def init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"初始化失败: {e}")
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # 创建双窗口显示：左边摄像头，右边皮影
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined[:, :width] = img  # 左边放原始摄像头画面
        
        if self.landmarker is None:
            return combined
        
        try:
            mp_image = mp_base.Image(image_format=mp_base.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                def get_point(landmark_id):
                    return (
                        int(landmarks[landmark_id].x * width) + width,  # 偏移到右边窗口
                        int(landmarks[landmark_id].y * height)
                    )
                
                # 提取关键点
                nose = get_point(0)
                left_shoulder = get_point(11)
                right_shoulder = get_point(12)
                left_elbow = get_point(13)
                right_elbow = get_point(14)
                left_wrist = get_point(15)
                right_wrist = get_point(16)
                left_hip = get_point(23)
                right_hip = get_point(24)
                left_knee = get_point(25)
                right_knee = get_point(26)
                left_ankle = get_point(27)
                right_ankle = get_point(28)
                
                # 计算中心点
                neck = (
                    (left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2
                )
                pelvis = (
                    (left_hip[0] + right_hip[0]) // 2,
                    (left_hip[1] + right_hip[1]) // 2
                )
                
                head_height = abs(neck[1] - nose[1]) * 0.8
                head_top = (nose[0], int(nose[1] - head_height))
                
                pose_data = {
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
                
                # 生成红色剪纸风格皮影
                shadow_img = create_red_paper_cut(pose_data, width, height)
                combined[:, width:] = shadow_img
        except Exception as e:
            print(f"检测错误: {e}")
        
        return combined

def create_red_paper_cut(pose_data, width, height):
    """创建红色剪纸风格皮影"""
    # 创建米色背景
    shadow_img = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    # 皮影风格绘制 - 红色
    limb_color = (0, 0, 255)  # 红色（BGR）
    thickness = 6
    
    data = pose_data
    
    # 绘制头部
    head_center = data.get('upper_neck', (width//2 + width, 100))
    head_radius = 30
    cv2.circle(shadow_img, head_center, head_radius, limb_color, -1)
    
    # 绘制身体
    neck = data.get('upper_neck')
    pelvis = data.get('pelvis')
    if neck and pelvis:
        cv2.line(shadow_img, neck, pelvis, limb_color, thickness + 4)
        
        left_shoulder = data.get('left_shoulder')
        right_shoulder = data.get('right_shoulder')
        if left_shoulder and right_shoulder:
            cv2.line(shadow_img, left_shoulder, right_shoulder, limb_color, thickness)
    
    # 绘制手臂
    if neck and data.get('left_shoulder') and data.get('left_elbow') and data.get('left_wrist'):
        cv2.line(shadow_img, neck, data['left_shoulder'], limb_color, thickness)
        cv2.line(shadow_img, data['left_shoulder'], data['left_elbow'], limb_color, thickness)
        cv2.line(shadow_img, data['left_elbow'], data['left_wrist'], limb_color, thickness)
    
    if neck and data.get('right_shoulder') and data.get('right_elbow') and data.get('right_wrist'):
        cv2.line(shadow_img, neck, data['right_shoulder'], limb_color, thickness)
        cv2.line(shadow_img, data['right_shoulder'], data['right_elbow'], limb_color, thickness)
        cv2.line(shadow_img, data['right_elbow'], data['right_wrist'], limb_color, thickness)
    
    # 绘制腿部
    if pelvis and data.get('left_hip') and data.get('left_knee') and data.get('left_ankle'):
        cv2.line(shadow_img, pelvis, data['left_hip'], limb_color, thickness)
        cv2.line(shadow_img, data['left_hip'], data['left_knee'], limb_color, thickness)
        cv2.line(shadow_img, data['left_knee'], data['left_ankle'], limb_color, thickness)
    
    if pelvis and data.get('right_hip') and data.get('right_knee') and data.get('right_ankle'):
        cv2.line(shadow_img, pelvis, data['right_hip'], limb_color, thickness)
        cv2.line(shadow_img, data['right_hip'], data['right_knee'], limb_color, thickness)
        cv2.line(shadow_img, data['right_knee'], data['right_ankle'], limb_color, thickness)
    
    # 绘制关节点
    joint_points = [
        data.get('head_top'), data.get('upper_neck'), data.get('nose'),
        data.get('left_shoulder'), data.get('right_shoulder'),
        data.get('left_elbow'), data.get('right_elbow'),
        data.get('left_wrist'), data.get('right_wrist'),
        data.get('pelvis'), data.get('left_hip'), data.get('right_hip'),
        data.get('left_knee'), data.get('right_knee'),
        data.get('left_ankle'), data.get('right_ankle')
    ]
    
    for point in joint_points:
        if point:
            cv2.circle(shadow_img, point, 8, limb_color, -1)
    
    return shadow_img

def get_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle if angle >= 0 else angle + 360

def rotate_bound(image, angle, key_point_y):
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

def create_shadow_puppet_online(pose_data, original_img=None):
    """在线版本：创建皮影效果"""
    try:
        # 创建黑色背景（皮影戏风格）
        height, width = original_img.shape[:2] if original_img is not None else (480, 640)
        shadow_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制半透明背景
        shadow_img[:] = (10, 10, 15)
        
        # 添加背景光效
        cv2.circle(shadow_img, (width//2, height//2), 200, (20, 20, 30), -1)
        
        data = pose_data
        
        # 皮影风格绘制 - 使用黄色和橙色
        limb_color = (255, 200, 50)  # 金黄色
        joint_color = (255, 100, 0)  # 橙红色
        thickness = 4
        
        # 绘制头部（圆形）
        head_center = data.get('upper_neck', (width//2, 100))
        head_radius = int(get_distences(data.get('head_top', (width//2, 50))[0], 
                                       data.get('head_top', (width//2, 50))[1],
                                       data.get('upper_neck', (width//2, 100))[0],
                                       data.get('upper_neck', (width//2, 100))[1]))
        head_radius = max(15, min(head_radius, 40))
        cv2.circle(shadow_img, head_center, head_radius, limb_color, -1)
        cv2.circle(shadow_img, head_center, head_radius, joint_color, 2)
        
        # 绘制身体（躯干）
        neck = data.get('upper_neck')
        pelvis = data.get('pelvis')
        if neck and pelvis:
            cv2.line(shadow_img, neck, pelvis, limb_color, thickness + 2)
            
            # 肩膀宽度
            left_shoulder = data.get('left_shoulder')
            right_shoulder = data.get('right_shoulder')
            if left_shoulder and right_shoulder:
                cv2.line(shadow_img, left_shoulder, right_shoulder, limb_color, thickness)
        
        # 绘制手臂
        # 左臂
        if neck and data.get('left_shoulder') and data.get('left_elbow') and data.get('left_wrist'):
            cv2.line(shadow_img, neck, data['left_shoulder'], limb_color, thickness)
            cv2.line(shadow_img, data['left_shoulder'], data['left_elbow'], limb_color, thickness)
            cv2.line(shadow_img, data['left_elbow'], data['left_wrist'], limb_color, thickness)
        
        # 右臂
        if neck and data.get('right_shoulder') and data.get('right_elbow') and data.get('right_wrist'):
            cv2.line(shadow_img, neck, data['right_shoulder'], limb_color, thickness)
            cv2.line(shadow_img, data['right_shoulder'], data['right_elbow'], limb_color, thickness)
            cv2.line(shadow_img, data['right_elbow'], data['right_wrist'], limb_color, thickness)
        
        # 绘制腿部
        # 左腿
        if pelvis and data.get('left_hip') and data.get('left_knee') and data.get('left_ankle'):
            cv2.line(shadow_img, pelvis, data['left_hip'], limb_color, thickness)
            cv2.line(shadow_img, data['left_hip'], data['left_knee'], limb_color, thickness)
            cv2.line(shadow_img, data['left_knee'], data['left_ankle'], limb_color, thickness)
        
        # 右腿
        if pelvis and data.get('right_hip') and data.get('right_knee') and data.get('right_ankle'):
            cv2.line(shadow_img, pelvis, data['right_hip'], limb_color, thickness)
            cv2.line(shadow_img, data['right_hip'], data['right_knee'], limb_color, thickness)
            cv2.line(shadow_img, data['right_knee'], data['right_ankle'], limb_color, thickness)
        
        # 绘制关节点（圆形标记）
        joint_points = [
            data.get('head_top'), data.get('upper_neck'), data.get('nose'),
            data.get('left_shoulder'), data.get('right_shoulder'),
            data.get('left_elbow'), data.get('right_elbow'),
            data.get('left_wrist'), data.get('right_wrist'),
            data.get('pelvis'), data.get('left_hip'), data.get('right_hip'),
            data.get('left_knee'), data.get('right_knee'),
            data.get('left_ankle'), data.get('right_ankle')
        ]
        
        for point in joint_points:
            if point:
                cv2.circle(shadow_img, point, 6, joint_color, -1)
                cv2.circle(shadow_img, point, 3, (255, 255, 0), -1)
        
        # 添加光晕效果
        for point in joint_points:
            if point:
                cv2.circle(shadow_img, point, 12, (50, 30, 0), -1)
        
        return shadow_img
        
    except Exception as e:
        print(f"生成皮影失败: {e}")
        if original_img is not None:
            return original_img
        return np.zeros((480, 640, 3), dtype=np.uint8)

def append_img_by_sk_points(img, append_img_path, key_point_y, first_point, second_point, 
                           append_img_reset_width=None, append_img_max_height_rate=1, 
                           middle_flip=False, append_img_max_height=None):
    try:
        if not append_img_path or not os.path.exists(append_img_path):
            return img
            
        append_image = cv2.imdecode(np.fromfile(append_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if append_image is None:
            return img
            
        sk_height = int(get_distences(first_point[0], first_point[1], second_point[0], second_point[1]) * append_img_max_height_rate)
        
        if append_img_max_height:
            sk_height = min(sk_height, append_img_max_height)
        
        sk_height = max(1, sk_height)
        
        if append_img_reset_width is None:
            sk_width = max(1, int(sk_height / append_image.shape[0] * append_image.shape[1]))
        else:
            sk_width = max(1, int(append_img_reset_width))
        
        if append_image.shape[0] > 0:
            key_point_y_new = int(key_point_y / append_image.shape[0] * sk_height)
        else:
            key_point_y_new = key_point_y
            
        append_image = cv2.resize(append_image, (sk_width, sk_height))
        
        img_height, img_width = img.shape[:2]
        
        if middle_flip:
            middle_x = int(img_width / 2)
            if first_point[0] < middle_x and second_point[0] < middle_x:
                append_image = cv2.flip(append_image, 1)
        
        angle = get_angle(first_point[0], first_point[1], second_point[0], second_point[1])
        append_image, move_x, move_y = rotate_bound(append_image, angle=angle, key_point_y=key_point_y_new)
        
        app_img_height, app_img_width = append_image.shape[:2]
        
        zero_x = first_point[0] - move_x
        zero_y = first_point[1] - move_y
        
        if len(append_image.shape) == 3:
            r_channel = append_image[:, :, 2]
            mask = (r_channel > 200) & (r_channel < 230)
            
            valid_rows = np.where(mask.any(axis=1))[0]
            if len(valid_rows) == 0:
                return img
            
            min_row, max_row = valid_rows[0], valid_rows[-1] + 1
            
            for i in range(min_row, max_row):
                valid_cols = np.where(mask[i])[0]
                if len(valid_cols) == 0:
                    continue
                
                target_y = zero_y + i
                if 0 <= target_y < img_height:
                    for j in valid_cols:
                        target_x = zero_x + j
                        if 0 <= target_x < img_width:
                            img[target_y][target_x] = append_image[i][j]
        
        return img
        
    except Exception as e:
        return img

def detect_pose(image):
    """姿态检测"""
    global _landmarker
    
    if not init_pose_landmarker():
        return None
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp_base.Image(image_format=mp_base.ImageFormat.SRGB, data=image_rgb)
        
        result = _landmarker.detect(mp_image)
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            height, width = image.shape[:2]
            
            def get_point(landmark_id):
                return (
                    int(landmarks[landmark_id].x * width),
                    int(landmarks[landmark_id].y * height)
                )
            
            nose = get_point(0)
            left_shoulder = get_point(11)
            right_shoulder = get_point(12)
            left_elbow = get_point(13)
            right_elbow = get_point(14)
            left_wrist = get_point(15)
            right_wrist = get_point(16)
            left_hip = get_point(23)
            right_hip = get_point(24)
            left_knee = get_point(25)
            right_knee = get_point(26)
            left_ankle = get_point(27)
            right_ankle = get_point(28)
            
            neck = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )
            pelvis = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            
            head_height = abs(neck[1] - nose[1]) * 0.8
            head_top = (nose[0], int(nose[1] - head_height))
            
            return {
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
    except Exception as e:
        st.error(f"检测错误: {e}")
    
    return None

def create_shadow_puppet(data, background_path='./background.jpg'):
    """生成皮影效果"""
    try:
        backgroup_image = cv2.imread(background_path)
        if backgroup_image is None:
            backgroup_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        image_flag = backgroup_image.copy()
        
        head_neck_dist = get_distences(data['head_top'][0], data['head_top'][1],
                                     data['upper_neck'][0], data['upper_neck'][1])
        min_width = max(10, int(head_neck_dist / 3))
        
        shoulder_width = get_distences(data['left_shoulder'][0], data['left_shoulder'][1],
                                     data['right_shoulder'][0], data['right_shoulder'][1])
        
        # 这里简化处理，实际项目需要素材文件
        # 绘制关键点用于演示
        for point in data.values():
            cv2.circle(image_flag, point, 5, (255, 0, 0), -1)
        
        return image_flag
        
    except Exception as e:
        st.error(f"生成错误: {e}")
        return None

# UI
st.title("🎭 实时皮影戏系统")
st.markdown("基于 MediaPipe 姿态识别")

mode = st.radio("选择模式", ["摄像头实时", "上传照片"])

if mode == "摄像头实时":
    st.info("请允许浏览器访问摄像头")
    webrtc_streamer(key="shadow-puppet", video_transformer_factory=ShadowPuppetTransformer)
else:
    uploaded_file = st.file_uploader("上传照片", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="原始图片", use_container_width=True)
        
        # 姿态检测
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 临时初始化检测器
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            temp_landmarker = vision.PoseLandmarker.create_from_options(options)
            
            mp_image = mp_base.Image(image_format=mp_base.ImageFormat.SRGB, data=cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            result = temp_landmarker.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                height, width = img_cv.shape[:2]
                
                # 绘制骨架
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28)
                ]
                
                img_with_pose = img_cv.copy()
                for id1, id2 in connections:
                    if id1 < len(landmarks) and id2 < len(landmarks):
                        pt1 = (int(landmarks[id1].x * width), int(landmarks[id1].y * height))
                        pt2 = (int(landmarks[id2].x * width), int(landmarks[id2].y * height))
                        cv2.line(img_with_pose, pt1, pt2, (0, 255, 0), 2)
                
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(img_with_pose, (x, y), 5, (255, 0, 0), -1)
                
                with col2:
                    st.image(cv2.cvtColor(img_with_pose, cv2.COLOR_BGR2RGB), caption="检测结果", use_container_width=True)
            else:
                st.warning("未检测到人体姿态")
        except Exception as e:
            st.error(f"检测失败: {e}")
    else:
        st.info("请上传包含人物的照片")
