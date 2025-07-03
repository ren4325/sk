from PIL import Image
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

data_path = "./edit/"
input_name_1 = "cropped_image_2.jpg"
input_name_2 = "cropped_image_3.jpg"
output_name_1 = "1_filtered_image.jpg"
output_name_2 = "2_filtered_image.jpg"
output_name_mask_1 =  "1_mask_image.jpg"
output_name_mask_2 =  "2_mask_image.jpg"

image_1 = read(data_path, input_name_1)
image_2 = read(data_path, input_name_2)

mask_img = Image.new("RGB", (800, 800), "white")
save(mask_img, "./mask/", "0_mask_image.jpg")
print("白い画像を 0_mask_image として保存しました。")

# 元画像のサイズ
original_width_1, original_height_1 = image_1.size
original_width_2, original_height_2 = image_2.size

new_width, new_height = 800, 800  # 新しい画像の幅と高さ

x, y = 0, 0  # 画像の左上の座標

x_shift, y_shift = 0,0
scale_factor = 1/1.534  # 拡大・縮小倍率
rotation_angle = -30.15 # 回転角度

center_x, center_y = (x + x + new_width) / 2 + + x_shift, (y + y + new_height) / 2 + y_shift# クロップ後の中心の座標

cropped_image_1 = crop_2(image_1, original_width_1, original_height_1, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift)
cropped_mask_img_1 = crop_2(mask_img, original_width_1, original_height_1, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift)

save(cropped_image_1, "./hdr/", output_name_1)
print("画像の一部を切り抜いて 1_filtered_image.jpg として保存しました。")
save(cropped_mask_img_1, "./mask/", output_name_mask_1)
print("白い画像に1_filtered_image.jpgと同様の変換をして、 1_mask_image.jpg として保存しました。")


x_shift_2, y_shift_2 = 0,0
scale_factor_2 = 1/2.073  # 拡大・縮小倍率
rotation_angle_2 = -14.85 # 回転角度

center_x_2, center_y_2 = (x + x + new_width) / 2 + + x_shift_2, (y + y + new_height) / 2 + y_shift_2# クロップ後の中心の座標

cropped_image_2 = crop_2(image_2, original_width_2, original_height_2, center_x_2, center_y_2, scale_factor_2, rotation_angle_2, x_shift_2, y_shift_2)
cropped_mask_img_2 = crop_2(mask_img, original_width_2, original_height_2, center_x_2, center_y_2, scale_factor_2, rotation_angle_2, x_shift_2, y_shift_2)

save(cropped_image_2, "./hdr/", output_name_2)
print("画像の一部を切り抜いて 2_filtered_image.jpg として保存しました。")
save(cropped_mask_img_2, "./mask/", output_name_mask_2)
print("白い画像に2_filtered_image.jpgと同様の変換をして、 2_mask_image.jpg として保存しました。")

