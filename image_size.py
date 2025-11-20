from PIL import Image
import numpy as np
import os

def read(data_path):
    # picture 内の最初の画像を読む
    files = os.listdir(data_path)
    images = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

    if len(images) == 0:
        raise FileNotFoundError("picture フォルダに画像がありません！")

    input_name = images[0]  # ←1枚だけ前提なのでこれでOK
    image = Image.open(os.path.join(data_path, input_name))

    return image, input_name  # ←元の名前も返す


def resize(image):
    width, height = image.size
    original_center_x, original_center_y = width / 2, height /2
    new_center = np.sqrt(width**2 + height**2) / 2

    box = (
        int(original_center_x - new_center),
        int(original_center_y - new_center),
        int(original_center_x + new_center),
        int(original_center_y + new_center)
    )

    resized_image = image.crop(box)
    return resized_image


def save(image, data_path, output_name):
    image.save(os.path.join(data_path, output_name))


# -------- メイン処理 --------

input_data_path = "picture"
output_data_path = "data"

# ★ファイル名を自動取得
image, input_name = read(input_data_path)

resized_image = resize(image)

# ★名前変えずに保存
save(resized_image, output_data_path, input_name)

print("保存完了")