import os
import cv2
import pandas as pd
import numpy as np

# 路径配置
IMAGE_DIR = './data/train-image/image/'
METADATA_PATH = './data/train-metadata.csv'
SAVE_DIR = './outputs_by_lesion/'
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取 metadata
df = pd.read_csv(METADATA_PATH)

# 图像拼接函数（带标题）
def create_collage(image_ids, label_name, width=256, height=256, font_scale=1.0, font_thickness=2):
    images = []
    for isic_id in image_ids:
        img_path = os.path.join(IMAGE_DIR, isic_id + '.jpg')
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (width, height))
            images.append(img_resized)
    if images:
        collage = cv2.hconcat(images)

        # 创建标题区域
        title_text = f"{label_name.capitalize()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(title_text, font, font_scale, font_thickness)[0]
        title_height = text_size[1] + 20
        title_img = np.ones((title_height, collage.shape[1], 3), dtype=np.uint8) * 255  # 白底

        # 居中绘制标题
        text_x = (collage.shape[1] - text_size[0]) // 2
        text_y = title_height - 10
        cv2.putText(title_img, title_text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        # 拼接标题和图像
        final_img = cv2.vconcat([title_img, collage])

        # 保存
        save_path = os.path.join(SAVE_DIR, f'{label_name}_collage.jpg')
        cv2.imwrite(save_path, final_img)
        print(f"✅ Saved collage with title: {save_path}")
    else:
        print(f"⚠️ No valid images found for {label_name}")

# 分别处理良性（target=0）和恶性（target=1）
for label in [0, 1]:
    label_name = 'benign' if label == 0 else 'malignant'
    subset = df[df['target'] == label]
    sampled_ids = subset.sample(n=5, random_state=42)['isic_id'].tolist()
    create_collage(sampled_ids, label_name)
