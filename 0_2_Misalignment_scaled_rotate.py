from PIL import Image
import os  # ← 追加

def read(data_path):
    # data_path 内のファイル一覧から、1枚目のファイルを読む
    files = os.listdir(data_path)
    if len(files) == 0:
        raise FileNotFoundError("data フォルダに画像ファイルがありません。")
    
    input_name = files[0]
    image = Image.open(data_path + input_name)

    return image


def crop_1(x, y , new_width, new_height, input_image):
    box_1 = (x, y, x + new_width, y + new_height) #左，上，右，下
    cropped_image_1 = input_image.crop(box_1) #クロップ

    return cropped_image_1

def crop_2(image, original_width, original_height, new_width, new_height, scale_factor, rotation_angle, x, y, x_shift, y_shift):
    
    new_center_x , new_center_y = x + x_shift + new_width /2, y + y_shift + new_height / 2
    
    rotated_image = image.rotate(rotation_angle, center=(new_center_x, new_center_y)) #回転

    new_size = (int(original_width * scale_factor), int(original_height * scale_factor)) 
    scaled_image = rotated_image.resize(new_size) #拡縮

    box_2 = ((new_center_x * scale_factor - new_width / 2),
             (new_center_y * scale_factor - new_height / 2),
             (new_center_x * scale_factor + new_width / 2),
             (new_center_y * scale_factor + new_height / 2)
            )
    cropped_image_2 = scaled_image.crop(box_2) 

    return cropped_image_2

def save(cropped_image, data_path, output_name):
    cropped_image.save(data_path + output_name)

data_path = "./data/"
op_data_path = "./before/"
output_name = "cropped_image_0001.png"

# ★ここ変更：ファイル名は自動取得
image = read(data_path)

original_width, original_height = image.size # 元画像のサイズ

new_width, new_height = original_width, original_height  # 新しい画像の幅と高さ

x, y = 0, 0  # 画像の左上の座標

cropped_image_1 = crop_1(x, y, new_width , new_height, image)

save(cropped_image_1, op_data_path, output_name)

for i in range(2, 11) :
    x_shift, y_shift = 0, 0
    scale_factor = 1.2 ** (i - 1)   # 拡大・縮小倍率
    rotation_angle = 3 * (i - 1)  # 回転角度

    cropped_image_2 = crop_2(image, original_width, original_height, new_width, new_height,
                             scale_factor, rotation_angle, x, y, x_shift, y_shift)

    save(cropped_image_2, op_data_path, f"cropped_image_{str(i).zfill(4)}.png")

print("画像を回転、拡大して cropped_image_0001〜0010.png として保存しました。")
