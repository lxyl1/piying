from rembg import remove
import os
from pathlib import Path

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 支持处理的图片格式
supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

if __name__ == '__main__':
    # 获取当前目录下所有图片文件
    for file in os.listdir('./shadow_play_material'):
        if file.lower().endswith(supported_formats):
            input_path = './shadow_play_material'+file
            output_path = output_dir / file  # 保持原文件名

            with open(input_path, 'rb') as i:
                with open(output_path, 'wb') as o:
                    print(f"正在处理: {input_path} => {output_path}")
                    input_data = i.read()
                    output_data = remove(input_data)
                    o.write(output_data)

    print("所有图片处理完成！输出目录:", output_dir.resolve())