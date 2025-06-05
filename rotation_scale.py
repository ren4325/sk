from PIL import Image

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

data_path = "./data/"
input_name = "cropped_image_2.jpg"
output_name = "filtered_image_1.jpg"

image = read(data_path, input_name)

original_width, original_height = image.size # 元画像のサイズ

new_width, new_height = 800, 800  # 新しい画像の幅と高さ

x, y = 0, 0  # 画像の左上の座標

x_shift, y_shift = 0,0
scale_factor = 1/1.534  # 拡大・縮小倍率
rotation_angle = -30.15 # 回転角度

center_x, center_y = (x + x + new_width) / 2 + + x_shift, (y + y + new_height) / 2 + y_shift# クロップ後の中心の座標

cropped_image = crop_2(image, original_width, original_height, center_x, center_y, scale_factor, rotation_angle, x_shift, y_shift)

save(cropped_image, data_path, output_name)

#HDR用
save(cropped_image, "./hdr/", output_name)



print("画像の一部を切り抜いて filtered_image.jpg として保存しました。")