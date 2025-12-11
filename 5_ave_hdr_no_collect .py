# ギャップ削除並列処理バージョン（位置合わせなし・マスクなし）
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import time
from concurrent.futures import ThreadPoolExecutor

# hdr合成
def image_fusion(image_list, c=100000, sigma=25):

    # FFT（周波数領域）に変換
    T_list = []
    for image in image_list:
        data = image.astype(np.float32)
        T = np.fft.fft2(data, axes=(0, 1))
        T_list.append(T)

    # 真ん中の画像を基準に差分を取る
    center_idx = len(T_list) // 2
    T_ref = T_list[center_idx]
    D_list = [T_ref - T for T in T_list]


    A_list = []
    for D in D_list:
        abs2 = np.abs(D) ** 2
        A = abs2 / (abs2 + c * (sigma ** 2))
        A_list.append(A)
    T_filter_list = [T + A * D for T, A, D in zip(T_list, A_list, D_list)]

    # 平均を取る
    fused_T = sum(T_filter_list) / len(T_filter_list)

    # 逆FFT
    output = np.real(np.fft.ifft2(fused_T, axes=(0, 1))).astype(np.float32)

    return np.clip(output, 0, 255)

# 平均合成
def average_fusion(image_array):

    if len(image_array) < 1:
        raise ValueError("画像リストが空です.")

    images = np.stack(image_array).astype(np.float32) 
    avg = np.mean(images, axis=0) 
    return avg.astype(np.float32)

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

    # ブロックごとの融合（image_fusion か average_fusion）
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
                    # 中心に近いほど重みが大きくなる
                    weight_y = 1 - abs(y - block_h // 2) / (block_h / 2) if block_h > 1 else 1.0
                    weight_x = 1 - abs(x - block_w // 2) / (block_w / 2) if block_w > 1 else 1.0
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
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(os.path.join(folder_path, f"{prefix}_frame_{i}.png"))

# メイン処理


# 実行時間計測開始
start_time = time.time()

# 出力フォルダ作成（位置合わせなし用）
results_base_path = os.path.join("result_no_collect")
results_hdr_path = os.path.join(results_base_path, "hdr_no_collect")
results_avg_path = os.path.join(results_base_path, "average_no_collect")
os.makedirs(results_hdr_path, exist_ok=True)
os.makedirs(results_avg_path, exist_ok=True)

# edit フォルダの画像をまとめて使う
edit_dir = "./edit"
image_files = [
    os.path.join(edit_dir, f)
    for f in os.listdir(edit_dir)
    if f.lower().endswith(".png")
]
image_files = sorted(image_files)

print(f"入力画像の枚数: {len(image_files)}")

# 画像を読み込む
image_list_pil = [Image.open(f) for f in image_files]

# 画像をテンソルに変換（もとのコードに合わせて torch 経由）
original_frames = torch.stack([torch.tensor(np.array(img)) for img in image_list_pil])

# numpy 配列（[H, W, C]）のリストにする
all_images = [img.numpy().astype(np.float32) for img in original_frames]

num_frames = len(all_images)

# --- セットごとに処理（中心 ±2 スライド）---
for set_idx in range(num_frames):
    center = set_idx
    start = max(0, center - 2)
    end = min(num_frames - 1, center + 2) + 1  # スライス用に +1

    group_images = all_images[start:end]

    print(f"[Set {set_idx}] 使用フレーム index: {list(range(start, end))}")

    # hdr合成
    fused_blocks = [fuse_blocks_parallel(group_images,
                                         block_size=(4000, 6000),
                                         overlap=(20, 20),
                                         fusion_method=image_fusion)]

    # 平均 合成
    average_image = [fuse_blocks_parallel(group_images,
                                          block_size=(4000, 6000),
                                          overlap=(20, 20),
                                          fusion_method=average_fusion)]

    # 保存
    set_name = f"{str(set_idx + 1).zfill(4)}"
    save_images(fused_blocks, results_hdr_path, f"nc_fused_blocks_set_{set_name}")
    save_images(average_image, results_avg_path, f"nc_average_image_set_{set_name}")

end_time = time.time()
print(f"全体の実行時間: {end_time - start_time:.2f} 秒")