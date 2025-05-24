import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

model = load_model("./models/model_kernel_5x5_epohs_15_lr_0.0005.keras")

img = cv.imread("./GTSRB/Training/00000/00000_00000.ppm")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (64, 64))
img = img.astype('float32') / 255.0
img_batch = np.expand_dims(img, axis=0)
pred = model.predict(img_batch)
print(np.argmax(pred))