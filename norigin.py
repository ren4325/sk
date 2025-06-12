# ギャップ削除並列処理バージョン
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import time
from concurrent.futures import ThreadPoolExecutor

# 画像融合の関数
# 画像融合の関数
def image_fusion(image_list, c=100000, sigma=25):
    T_list = []
    for image in image_list:
        data = np.array(image).astype(np.float32)
        T = np.fft.fft2(data, axes=(0, 1))
        T_list.append(T)

    # 黒除外マスク（全チャネルでしきい値10以上を情報ありとする）
    masks = [np.any(img > 10, axis=-1, keepdims=True).astype(np.float32) for img in image_list]
    valid_mask = np.prod(masks, axis=0)

    D_list = [T_list[0] - T for T in T_list]
    A_list = [(np.abs(D) ** 2) / (np.abs(D) ** 2 + c * sigma ** 2) for D in D_list]
    T_filter_list = [T + A * D for T, A, D in zip(T_list, A_list, D_list)]

    fused_T = sum(T_filter_list) / len(T_filter_list)
    fused_img = np.abs(np.fft.ifft2(fused_T, axes=(0, 1))).astype(np.float32)

    # 欠損部を1枚目の画像で補完（必要なら他の平均でもOK）
    fallback = image_list[0]
    output = fused_img * valid_mask + fallback * (1 - valid_mask)

    return np.clip(output, 0, 255)


# 平均化による画像融合
def average_fusion(image_array):
    if len(image_array) < 1:
        raise ValueError("画像リストが空です.")
    return np.mean(image_array, axis=0)

# 並列処理用のブロック融合関数
def process_block(params):
    i, j, block_size, overlap, h, w, image_list, fusion_method = params
    stride_y = block_size[0] - overlap[0]
    stride_x = block_size[1] - overlap[1]

    # ブロック範囲を計算
    y_end = min(i + block_size[0], h)
    x_end = min(j + block_size[1], w)

    # 各ブロックを取得
    block_list = [img[i:y_end, j:x_end, :] for img in image_list]

    # ブロックごとの融合
    fused_block = fusion_method(block_list)

    return i, j, y_end, x_end, fused_block

# 並列処理を使用したブロック融合
def fuse_blocks_parallel(image_list, block_size=(4000, 6000), overlap=(20, 20), fusion_method=image_fusion):
    if len(image_list) < 1:
        raise ValueError("画像リストが空です.")

    h, w, _ = image_list[0].shape
    fused_image = np.zeros_like(image_list[0], dtype=np.float32)
    weight_image = np.zeros_like(image_list[0], dtype=np.float32)

    stride_y = block_size[0] - overlap[0]
    stride_x = block_size[1] - overlap[1]

    tasks = []
    for i in range(0, h, stride_y):
        for j in range(0, w, stride_x):
            tasks.append((i, j, block_size, overlap, h, w, image_list, fusion_method))

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_block, tasks)

    for i, j, y_end, x_end, fused_block in results:
        block_h, block_w, _ = fused_block.shape
        y_start, x_start = max(0, i), max(0, j)
        for y in range(block_h):
            for x in range(block_w):
                if y_start + y < h and x_start + x < w:
                    weight_y = 1 - abs(y - block_h // 2) / (block_h / 2)
                    weight_x = 1 - abs(x - block_w // 2) / (block_w / 2)
                    weight = weight_y * weight_x
                    fused_image[y_start + y, x_start + x, :] += fused_block[y, x, :] * weight
                    weight_image[y_start + y, x_start + x, :] += weight

    weight_image[weight_image == 0] = 1
    fused_image = fused_image / weight_image

    return fused_image.astype(np.uint8)

# 画像の保存
def save_images(images, folder_path, prefix):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(folder_path, f"{prefix}_frame_{i}.png"))

# 実行時間計測開始
start_time = time.time()

# 画像ファイルのパスを指定する
image_files = [
    './hdr/cropped_image_1.jpg',   # これが補完元（image_list[0]）
    './hdr/filtered_image_1.jpg'   # 欠損あり（image_list[1]）
]

# 入力画像の枚数を表示
print(f"入力画像の枚数: {len(image_files)}")

# 画像を読み込む
image_list = [Image.open(f) for f in image_files]
    
# 画像をテンソルに変換
original_frames = torch.stack([torch.tensor(np.array(img)) for img in image_list])

# フレームの融合（並列処理）
fused_blocks = [fuse_blocks_parallel([img.numpy() for img in original_frames], fusion_method=image_fusion)]
average_image = [fuse_blocks_parallel([img.numpy() for img in original_frames], fusion_method=average_fusion)]


# 保存処理
save_images(fused_blocks, 'results', 'fused_blocks')
save_images(average_image, 'results', 'average_image')


end_time = time.time()
print(f"全体の実行時間: {end_time - start_time:.2f} 秒")