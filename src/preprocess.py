import numpy as np
from PIL import Image
from .settings import USE_VGG16_PREPROCESS, TARGET_SIZE

def load_and_preprocess_image(file, target_size=TARGET_SIZE):
    img = Image.open(file).convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype("float32")

    if USE_VGG16_PREPROCESS:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr /= 255.0

    return img, np.expand_dims(arr, axis=0)
