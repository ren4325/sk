from PIL import Image
import os
import numpy as np

def read(data_path, input_name):
    image = Image.open(data_path + input_name)

    return image


def crop_2(image, original_width, original_height, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift ):
    #rotated_image = image.rotate(rotation_angle, center=(center_x, center_y)) #回転
    
    new_size = (int(original_width * scale_factor), int(original_height * scale_factor))
    scaled_image = image.resize(new_size) #拡縮


    box_2 = (
        round((center_x * scale_factor - new_width / 2) ),
        round((center_y * scale_factor - new_height / 2) ),
        round((center_x * scale_factor - new_width / 2) + new_width ),
        round((center_y * scale_factor - new_height / 2) + new_height)
        )
    
    cropped_image = scaled_image.crop(box_2)
    cropped_image = cropped_image.rotate(rotation_angle, center=(center_x, center_y))
    
    return cropped_image

def save(cropped_image, data_path, output_name):
    cropped_image.save(data_path + output_name)

# 白画像の作成
mask_img = Image.new("RGB", (3024, 3024), "white")

# データパスの指定
input_path = "./edit/"
output_path = "./hdr/"
mask_path =  "./mask/"

# フォルダ内のファイル一覧を取得（画像だけ残してソート）
input_images = sorted(
    f for f in os.listdir(input_path)
)
num_images = len(input_images)

for i, input_image_name in enumerate(input_images, start=1):
    print(f"{i}[枚目の画像({input_image_name})")
    image = read(input_path, input_image_name)

    # 元画像のサイズ
    original_width, original_height = image.size

    # 新しい画像の幅と高さ
    new_width, new_height = 2000, 2000

    # 画像の左上の座標
    x, y = 0, 0

    x_shift, y_shift = 0,0
    scale_factor = 1 / float(input("倍率"))  # 拡大・縮小倍率
    rotation_angle = float(input("回転")) # 回転角度

    # クロップ後の中心の座標
    center_x, center_y = (x + x + new_width) / 2 + + x_shift, (y + y + new_height) / 2 + y_shift

    cropped_image_1 = crop_2(image, original_width, original_height, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift)
    cropped_mask_img_1 = crop_2(mask_img, original_width, original_height, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift)

    save(cropped_image_1, "./hdr/", f"{i}_filtered_image.jpg")
    print(f"画像の一部を切り抜いて {i}_filtered_image.jpg として保存しました。")
    save(cropped_mask_img_1, "./mask/", f"{i}_mask_image.jpg")
    print(f"白い画像に{i}_filtered_image.jpgと同様の変換をして、 {i}_mask_image.jpg として保存しました。")