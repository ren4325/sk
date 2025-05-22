from PIL import Image

path = "./data/u02_MusicBox_2K.jpg" #画像のある場所を入力
img = Image.open(path).convert("RGB")

x, y = 1920, 1080 #変更後のサイズ
new_size = (x, y)

resized_img = img.resize(new_size)

resized_img.save("./picture/input.jpg") #保存する先を入力