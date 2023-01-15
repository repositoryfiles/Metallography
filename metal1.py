import datetime
import os
import sys
import tkinter
from tkinter import filedialog

import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("c:/python/keras_Model.h5", compile=False)

# Load the labels
class_names = open("c:/python/labels.txt", "r").readlines()
# 改行コードを取り除く
class_names = list(map(lambda s: s.rstrip("\n"), class_names))

iDir = "c:/data"

# 画像ファイル選択
root = tkinter.Tk()
root.withdraw()
fTyp = [
    ("jpg", "*.jpg"),
    ("BMP", "*.bmp"),
    ("png", "*.png"),
    ("tiff", "*.tif"),
]
fname = filedialog.askopenfilenames(filetypes=fTyp, initialdir=iDir)

# 画像ファイルを選ばなかったときの処理（プログラム終了）
if fname == "":
    sys.exit()

class_name = []
confidence_score = []

i = 0
for f in fname:
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(f).convert("RGB")

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name.append(class_names[index])
    confidence_score.append(prediction[0][index])
    i += 1

now = datetime.datetime.now()

with open(
    str(os.path.dirname(fname[0]))
    + "/result_{0:%Y%m%d%H%M}".format(now)
    + ".csv",
    "w",
    encoding="utf-8",
) as f1:
    i = 0
    for f in fname:
        print(f"{f}, {class_name[i]}, (score:{confidence_score[i]:.2f})")
        f1.write(
            "{}, {}, {:.2f}\n".format(
                str(f), class_name[i], confidence_score[i]
            )
        )
        i += 1
