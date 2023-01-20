import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir

# GLOBAL PARAMETERS
H = 512
W = 512

if __name__ == "__main__":
    # SEED NUMPY
    np.random.seed(42)
    tf.random.set_seed(42)

    # DIR FOR STORING FILES
    create_dir("test_images/mask")

    # LOAD MODEL
    with CustomObjectScope({'iou':iou, 'dice_coef':dice_coef, 'dice_loss':dice_loss}):
        model = keras.models.load_model("files/model.h5")

    # LOAD DATASET
    data_x = glob("test_images/image/*")

    for path in tqdm(data_x, total=len(data_x)):
        # EXTRACT NAME
        name = path.split("\\")[-1].split(".")[0]

        # READ THE IMAGE
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        # PREDICTIONS
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        # SAVE IMAGE
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imwrite(f"test_images/mask/{name}.png", cat_images)