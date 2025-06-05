from PIL import Image

def overlay_images(img_path1, img_path2, output_path):
    # 画像を読み込み
    img1 = Image.open(img_path1).convert("RGBA")
    img2 = Image.open(img_path2).convert("RGBA")

    # img2 のアルファ値を 50% に設定
    img2_with_alpha = img2.copy()
    img2_with_alpha.putalpha(128)

    # 背景に img1 を使って img2 を重ねる
    img1.paste(img2_with_alpha, (0, 0), img2_with_alpha)

    # 保存
    img1.save(output_path)

# 使い方例
overlay_images("data/cropped_image_1.jpg", "data/filtered_image_1.jpg", "data/overlay_output.png")

print("cropped_image_1.jpgとfiltered_image_1.jpgを重ねてdata/overlay_output.pngとして保存しました。")