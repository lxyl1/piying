# 实时皮影戏系统

基于 MediaPipe 姿态识别的实时皮影戏效果系统。

## 功能特点

- 实时摄像头捕捉人体姿态
- 自动映射到皮影角色关节
- 流畅的皮影动画效果

## 环境要求

- Python 3.8 - 3.12
- 摄像头

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/lxyl1/piying.git
cd piying
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型文件

从以下地址下载 MediaPipe 姿态检测模型：

```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "pose_landmarker_lite.task"

# Linux/Mac
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

将 `pose_landmarker_lite.task` 放在项目根目录。

### 4. 准备素材

确保 `shadow_play_material/` 目录下有以下图片：
- head.jpg - 头部
- body.jpg - 身体
- right_hip.jpg, left_hip.jpg - 大腿
- right_knee.jpg, left_knee.jpg - 小腿
- right_elbow.jpg, left_elbow.jpg - 上臂
- right_wrist.jpg, left_wrist.jpg - 前臂
- background.jpg - 背景图

## 运行程序

### 方式一：Web 版本（推荐）

```bash
streamlit run app.py
```

然后在浏览器打开显示的链接，上传照片即可看到皮影效果。

### 方式二：桌面版本（需要摄像头）

```bash
python 1.py
```

#### 操作说明

- **q** - 退出程序
- 左侧窗口显示摄像头画面
- 右侧窗口显示皮影效果

## 性能优化

如果运行卡顿，可以：

1. 降低摄像头分辨率（修改代码第 325-326 行）
2. 增加跳帧数（修改 `process_every_n_frames` 变量）
3. 关闭其他占用 CPU 的程序

## 项目结构

```
py_project/
├── 1.py                      # 主程序
├── requirements.txt          # 依赖包
├── pose_landmarker_lite.task # MediaPipe模型（需下载）
├── background.jpg            # 背景图
└── shadow_play_material/     # 皮影素材
    ├── head.jpg
    ├── body.jpg
    ├── right_hip.jpg
    ├── left_hip.jpg
    ├── right_knee.jpg
    ├── left_knee.jpg
    ├── right_elbow.jpg
    ├── left_elbow.jpg
    ├── right_wrist.jpg
    └── left_wrist.jpg
```

## 技术栈

- **MediaPipe** - 姿态识别
- **OpenCV** - 图像处理
- **NumPy** - 数值计算
- **Pillow** - 图像操作

## 许可证

MIT License
