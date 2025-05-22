from PIL import Image

def read(data_path, input_name):
    image = Image.open(data_path + input_name)

    return image


def crop(image, new_width, new_height, x_shift, y_shift):
    box = (x_shift, y_shift, x_shift + new_width, y_shift + new_height)
    cropped_image = image.crop(box)

    return cropped_image

def save(cropped_image, data_path, output_name):
    cropped_image.save(data_path + output_name)

data_path = "./data/"
input_name = "filtered_image_1.jpg"
output_name = "filtered_image_2.jpg"

image = read(data_path, input_name)

original_width, original_height = image.size # 元画像のサイズ

new_width, new_height = 500, 500  # 新しい画像の幅と高さ

x_shift, y_shift = -103, -46

cropped_image = crop(image, new_width, new_height, x_shift, y_shift)

save(cropped_image, data_path, output_name)


print("画像の一部を切り抜いて filtered_image.jpg として保存しました。")