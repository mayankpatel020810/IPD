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
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data

# GLOBAL PARAMETERS
H = 512
W = 512

# CREATING A DIRECTORY
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) *128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    # SEED NUMPY
    np.random.seed(42)
    tf.random.set_seed(42)

    # DIR FOR STORING FILES
    create_dir("results")

    # LOAD MODEL
    with CustomObjectScope({'iou':iou, 'dice_coef':dice_coef, 'dice_loss':dice_loss}):
        model = keras.models.load_model("files/model.h5")

    # LOAD TEST SET
    dataset_path = "new_data"
    valid_path = os.path.join(dataset_path, "test")
    test_x, test_y = load_data(valid_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    # EVALUATION
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # EXTRACT NAME
        name = x.split("\\")[-1].split(".")[0]

        # READ IMAGE
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        # READ MASK
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        # PREDICTION
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        # SAVING THE PREDICTION
        save_image_path = f"results/{name}.png"
        save_results(image, mask, y_pred, save_image_path)

        # FLATTEN THE ARRAY
        mask = mask.flatten()
        y_pred = y_pred.flatten()

        # CALCULATE THE MATRIX VALUES
        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
        jacc_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jacc_value, recall_value, precision_value])

    # METRIC VALUES
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy","F1","Jaccard","Recall","Precision"])
    df.to_csv("files/score.csv")