from PIL import Image
import os
import numpy as np
import csv

def read(data_path, input_name):
    image = Image.open(data_path + input_name)

    return image

# csv読み込み
def load_csv_params(csv_path="estimated_value.csv"):

    params = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        estimated_value = list(csv.reader(f))

    # 2行目:ヘッダー想定 3行目以降:データ
    for row in estimated_value[2:]:
        if len(row) < 8:
            continue

        set_num, base_name, filename, index, sx, sy, s, rot = row
        
        params[(int(set_num), filename)] = {
            "base_name": base_name,
            "index": int(index),
            "shift_x": float(sx),
            "shift_y": float(sy),
            "scale_factor": float(s),
            "rotation_angle": float(rot),
        }

    return params


def crop_2(image, original_width, original_height, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift ):
    
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
    save_path = os.path.join(data_path, output_name)
    cropped_image.save(save_path)

# 白画像の作成
mask_img = Image.new("RGB", (3024, 3024), "white")

# データパスの指定
input_path = "./edit/"
output_path = "./hdr/"
mask_path =  "./mask/"

# 処理の分割の指定
hdr_range = 3

# フォルダ内のファイル一覧を取得（画像だけ残してソート）
input_images = sorted(os.listdir(input_path))  

#csvファイルの読み込み
csv_params = load_csv_params("estimated_value.csv")


for i, input_image_name in enumerate(input_images, start=1):
    
    # グループ決定
    start = max(1, i - hdr_range//2)
    end = min(len(input_images), i + hdr_range//2)
    group = input_images[start-1:end]

    # フォルダ作成
    set_folder = os.path.join(output_path, f"hdr_set_{str(i).zfill(4)}")
    mask_folder = os.path.join(mask_path, f"mask_set_{str(i).zfill(4)}")
    os.makedirs(set_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    for g_idx, g_name in enumerate(group, start=1):
        image = read(input_path, g_name)
        original_width, original_height = image.size
        new_width, new_height = original_width, original_height
        x, y = 0, 0


        # CSVからset番号をキーにして取り出す
        p = csv_params[(i, g_name)]
        scale_factor   = p["scale_factor"]
        rotation_angle = p["rotation_angle"]
        x_shift = p["shift_x"]
        y_shift = p["shift_y"]

        print(f"CSVファイルの値を適用: set={i}, file={g_name}, scale={scale_factor}, rot={rotation_angle}°")


        # クロップ後の中心の座標
        center_x, center_y = (x + x + new_width) / 2 + + x_shift, (y + y + new_height) / 2 + y_shift

        cropped_image = crop_2(image, original_width, original_height, center_x, center_y, 1/scale_factor, rotation_angle, x_shift, y_shift)
        cropped_mask_img = crop_2(mask_img, original_width, original_height, center_x, center_y, 1/scale_factor, rotation_angle, x_shift, y_shift)

        save(cropped_image, set_folder, f"{g_idx}_{g_name}_filtered.jpg")
        save(cropped_mask_img, mask_folder, f"{g_idx}_{g_name}_mask.jpg")
    
print(f"hdr_set_{i} を作成")
