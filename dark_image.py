from PIL import Image
import numpy as np

def save(image, data_path, output_name):
    image.save(data_path + output_name)

# 正しく画像を読み込んでからNumPy化
image = Image.open("./hdr/cropped_image_1.jpg")
image_np = np.array(image).astype(np.float32)

# 暗くする（明度30%）
dark_image_np = (image_np * 0.3).clip(0, 255).astype(np.uint8)

# PILに戻す
dark_image = Image.fromarray(dark_image_np)

# 保存
save(dark_image, "./hdr/", "cropped_image_1.jpg")
print("cropped_image_1.png を暗くしました。")
