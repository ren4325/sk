from PIL import Image
import numpy as np
import os

# 入出力フォルダ
input_path = "./before/"
output_path = "./edit/"
os.makedirs(output_path, exist_ok=True)

# フォルダ内の画像を処理
for file in os.listdir(input_path):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # 画像を読み込み → NumPy配列に変換
    img = Image.open(os.path.join(input_path, file)).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # ガウスノイズを生成（平均, 標準偏差）
    noise = np.random.normal(0, 10, arr.shape)

    # 画像にノイズを足してクリップ
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # 画像に戻して保存
    noisy_img = Image.fromarray(noisy_arr)
    noisy_img.save(os.path.join(output_path, file))

print("ノイズ付き画像を保存しました")