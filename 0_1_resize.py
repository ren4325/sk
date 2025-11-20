from PIL import Image
import os

# 入力フォルダと出力フォルダ
input_folder = "./picture"
output_folder = "./data"

def read_single_image(folder):
    # フォルダ内の最初の画像を読む（1枚だけ前提）
    files = os.listdir(folder)
    images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    if len(images) == 0:
        raise FileNotFoundError("picture フォルダに画像がありません。")

    image_path = os.path.join(folder, images[0])
    img = Image.open(image_path).convert("RGB")
    return img


# ---- メイン処理 ----

# ① picture から1枚読み込み
img = read_single_image(input_folder)

# ② リサイズ後のサイズ
new_size = (2000, 2000)

# ③ リサイズ
resized_img = img.resize(new_size)

# ④ data/resized.jpg として保存
output_path = os.path.join(output_folder, "resized.jpg")
resized_img.save(output_path)

print("resized.jpg を data フォルダに保存しました")
