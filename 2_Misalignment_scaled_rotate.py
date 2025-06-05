from PIL import Image

def read(data_path, input_name):
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
input_name = "sample.jpg"
output_name_1 = "cropped_image_1.jpg"
output_name_2 = "cropped_image_2.jpg"

image = read(data_path, input_name)

original_width, original_height = image.size # 元画像のサイズ

new_width, new_height = 800, 800  # 新しい画像の幅と高さ

x, y = 700, 900  # 画像の左上の座標

x_shift, y_shift = 0, 0
scale_factor =1.5  # 拡大・縮小倍率
rotation_angle = 30  # 回転角度

cropped_image_1 = crop_1(x, y, new_width , new_height, image)
cropped_image_2 = crop_2(image, original_width, original_height, new_width, new_height, scale_factor, rotation_angle, x, y, x_shift, y_shift)

save(cropped_image_1, data_path, output_name_1)
save(cropped_image_2, data_path, output_name_2)

#HDR用
save(cropped_image_1, "./hdr/", output_name_1)



print("画像の一部を切り抜いて cropped_image.jpg として保存しました。")