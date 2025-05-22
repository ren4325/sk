from PIL import Image
import numpy as np

def amplitude_spectrum(image_path):
    # 画像を読み込み
    image = Image.open(image_path).convert("L")
    # グレースケールに変換
    
    # 画像をNumPy配列に変換
    img_array = np.array(image)
    
    # 2次元フーリエ変換を実施
    fft_result = np.fft.fft2(img_array)
    fft_shift = np.fft.fftshift(fft_result)  # ゼロ周波数成分を中心に移動
    
    # 振幅スペクトルを計算
    amplitude = np.abs(fft_shift)
    amplitude = np.log(1 + amplitude)  # 可視化のために対数スケールを適用
    
    return amplitude


def save(image_array, base_array, output_path):
    # 振幅スペクトルを0-255の範囲にスケーリングして保存
    scaled_image = np.uint8(255 * (image_array / np.max(base_array)))
    img = Image.fromarray(scaled_image)
    img.save(output_path)

# 入力画像と出力先のパス
image1_path = "data/cropped_image_1.jpg"  # 画像1のパスに置き換えてください
image2_path = "data/cropped_image_1.jpg"  # 画像2のパスに置き換えてください
output1_path = "data/spectrum1.jpg"
output2_path = "data/spectrum2.jpg"

# 1枚目のスペクトルを計算して保存
ref_spectrum_1 = amplitude_spectrum(image1_path)
ref_spectrum_2 = amplitude_spectrum(image2_path)

save(ref_spectrum_1, ref_spectrum_1, output1_path)
save(ref_spectrum_2, ref_spectrum_1, output2_path)


# save_spectrum_with_reference(image1_path, ref_spectrum, output1_path)

# 2枚目のスペクトルをリファレンス基準で正規化して保存
# save_spectrum_with_reference(image2_path, ref_spectrum, output2_path)
