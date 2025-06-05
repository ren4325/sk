from PIL import Image
import numpy as np
import scipy.signal
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def size(image_array):
    source_height, source_width = image_array.shape  # 元画像のサイズ
    center_x, center_y = (source_width / 2, source_height / 2)
    return source_width, source_height, center_x, center_y


def gaussian_2d(size, sigma_x=1.0, sigma_y=1.0):
    """
    2次元ガウス窓を生成する関数。
    
    Parameters
    ----------
    size : int or (int,int)
        ガウス窓の大きさ。1つの整数を与えた場合は正方窓、(height,width)で指定可能。
    sigma_x : float
        x方向(幅方向)の標準偏差
    sigma_y : float
        y方向(高さ方向)の標準偏差

    Returns
    -------
    g : 2D numpy array
        指定サイズの2次元ガウス窓
    """
    if isinstance(size, int):
        size = (size, size)
    h, w = size

    # 画素座標
    y, x = np.indices((h, w))
    
    # ガウス分布の中心を窓の中央に設定
    x_mean = (w - 1) / 2.0
    y_mean = (h - 1) / 2.0
    
    # 共分散行列（対角行列）
    cov = [[sigma_y**2, 0],
           [0, sigma_x**2]]

    # 2次元ガウス分布オブジェクトの作成
    rv = multivariate_normal(mean=[y_mean, x_mean], cov=cov)

    # 各画素に対してPDF(確率密度関数)を計算
    g = rv.pdf(np.dstack((y,x)))
    g /= np.max(g)  # 正規化

    return g


def magnitude(input_image):
    # フーリエ変換前に窓関数（ガウス窓）をかける
    window = gaussian_2d(input_image.shape, sigma_x=input_image.shape[1]/5, sigma_y=input_image.shape[0]/5)
    fft_image = np.fft.fft2(input_image * window)  # 2次元フーリエ変換，ガウス窓を適用
    fft_image_shifted = np.fft.fftshift(fft_image)  # 中心をシフト
    amplitude_spectrum = np.abs(fft_image)    
    amplitude_spectrum_shifted = np.abs(fft_image_shifted)
    magnitude_image = np.log(np.abs(amplitude_spectrum) + 1)  
    magnitude_shifted_image = np.log(np.abs(amplitude_spectrum_shifted) + 1)  
    """
    image_array = input_image * window
    scaled_image = np.uint8(255 * ((image_array - np.min(image_array)) / (np.max(image_array) -np.min(image_array))))
    img = Image.fromarray(scaled_image)
    img.save("./data/gaussian.jpg")
    """

    return amplitude_spectrum, amplitude_spectrum_shifted, magnitude_image, magnitude_shifted_image


def log_polar_transform(magnitude, center_x, center_y, source_height, source_width):
    max_radius = np.minimum(center_x, center_y)
    log_radius = np.linspace(0, np.log(max_radius), source_height)
    theta = np.linspace(0, 2 * np.pi, source_width)
    
    # for i, lr in enumerate(log_radius):
    #     for j, t in enumerate(theta):
    #         radius = np.exp(lr)
    #         x = int(center_x + radius * np.cos(t))
    #         y = int(center_y + radius * np.sin(t))
    #         if 0 <= x < source_width and 0 <= y < source_height:
    #             transformed_array[i, j] = magnitude[y, x]

    # x, yを整数に丸めず，map_coordinatesで画素値を補間（「補間」については調べてみてください）
    # map_coordinatesは、(row, col) = (y, x)の順で座標を渡す必要があるので注意
    r_exp = np.exp(log_radius)
    rr, tt = np.meshgrid(r_exp, theta, indexing='ij')
    x_coords = center_x + rr * np.cos(tt)
    y_coords = center_y + rr * np.sin(tt)
    coords = [y_coords.ravel(), x_coords.ravel()]
    values = map_coordinates(magnitude, coords, order=3, mode='nearest')
    transformed_array = values.reshape((source_height, source_width))
    
    return transformed_array


def cross_power(transformed_array_1, transformed_array_2):
    transformed_array_1 -= np.mean(transformed_array_1)
    transformed_array_2 -= np.mean(transformed_array_2) #平均を０に
    fft_amplitude_1 = np.fft.fft2(transformed_array_1, s=transformed_array_1.shape)
    fft_amplitude_2 = np.fft.fft2(transformed_array_2, s=transformed_array_2.shape)

    # 分母が0にならないように最小値を設定
    denominator = np.abs(fft_amplitude_1 * np.conj(fft_amplitude_2))
    denominator[denominator < 1e-10] = 1e-10  # 小さな値を安全な最小値に置き換え

    cross_power_spectrum = (fft_amplitude_1 * np.conj(fft_amplitude_2)) / denominator
    cross_corr = np.fft.ifft2(cross_power_spectrum)
    cross_corr = np.fft.fftshift(cross_corr)  # 中心にシフト
    cross_corr = np.real(cross_corr)  # 実部のみを取得
    save(cross_corr, "./data/", "cross_corr.jpg")

    # 最大値のインデックスを取得
    max_index = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    shift_y, shift_x = max_index[0] - (cross_corr.shape[0] // 2), max_index[1] - (cross_corr.shape[1] // 2)

    return shift_x, shift_y


def rho_theta(magnitude_1, shift_x, shift_y, center_x, center_y):
    scale_factor = np.exp(shift_y * np.log(np.hypot(center_x, center_y)) / magnitude_1.shape[0])
    rotation_angle =-1 *( (shift_x * 360 / magnitude_1.shape[1]) % 360)  # 角度正規化

    return scale_factor, rotation_angle


def save(image_array, data_path, output_name):
    # 振幅スペクトルを0-255の範囲にスケーリングして保存
    scaled_image = np.uint8(255 * ((image_array - np.min(image_array)) / (np.max(image_array) -np.min(image_array))))
    img = Image.fromarray(scaled_image)
    img.save(data_path + output_name)

# データパスとファイル名の設定
data_path = "./data/"
input_name_1 = 'cropped_image_1.jpg'
input_name_2 = 'cropped_image_2.jpg'

output_name_1 = "log_polar_1.jpg"
output_name_2 = "log_polar_2.jpg"

# 画像を読み込み
input_image_1 = Image.open(data_path + input_name_1).convert("L")
input_image_2 = Image.open(data_path + input_name_2).convert("L")

# numpy配列へ変換
image_array_1 = np.array(input_image_1)
image_array_2 = np.array(input_image_2)

# サイズ，中心を測る
source_width, source_height, center_x, center_y = size(image_array_1)

# 振幅スペクトルへ変換
magnitude_1, magnitude_shifted_1, magnitude_image_1, magnitude_shifted_image_1 = magnitude(image_array_1)
magnitude_2, magnitude_shifted_2, magnitude_image_2, magnitude_shifted_image_2 = magnitude(image_array_2)


# 振幅スペクトルの画像として保存
save(magnitude_image_1, data_path, "magnitude_1.jpg")
save(magnitude_image_2, data_path, "magnitude_2.jpg")
save(magnitude_shifted_image_1, data_path, "magnitude_shifted_1.jpg")
save(magnitude_shifted_image_2, data_path, "magnitude_shifted_2.jpg")

# 対数極座標へ変換
transformed_array_1 = log_polar_transform(magnitude_shifted_1, center_x, center_y, source_height, source_width)
transformed_array_2 = log_polar_transform(magnitude_shifted_2, center_x, center_y, source_height, source_width)

transformed_image_1 = log_polar_transform(magnitude_shifted_image_1, center_x, center_y, source_height, source_width)
transformed_image_2 = log_polar_transform(magnitude_shifted_image_2, center_x, center_y, source_height, source_width)

save(transformed_image_1, data_path, "log_polar_magnitude_1.jpg")
save(transformed_image_2, data_path, "log_polar_magnitude_2.jpg")


# シフト量の推定
shift_x, shift_y = cross_power(transformed_array_1, transformed_array_2)
scale_factor, rotation_angle = rho_theta(magnitude_1, shift_x, shift_y, center_x, center_y)

print(f"Scale factor (2 に対して): {round(scale_factor, 3)}")
print(f"Rotation angle (2 に対して): {round(rotation_angle, 3)} degrees")