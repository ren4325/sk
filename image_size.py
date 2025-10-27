from PIL import Image
import numpy as np

def read(data_path, input_name):
    image = Image.open(data_path + "/" + input_name)

    return image

def resize(image):
    
    width, height = image.size
    original_center_x, original_center_y = width / 2, height /2
    new_center = np.sqrt(width**2 + height**2) / 2

    box=(int(original_center_x - new_center),
         int(original_center_y - new_center),
         int(original_center_x + new_center),
         int(original_center_y + new_center)
        )

    resized_image = image.crop(box)

    return resized_image

def save(image, data_path, output_name):
    image.save(data_path +"/" + output_name)


input_data_path = "picture"
output_data_path = "data"
input_name = "input.jpg"
output_name = "sample.jpg"

image = read(input_data_path, input_name)
resized_image = resize(image)

save(resized_image, output_data_path, output_name)