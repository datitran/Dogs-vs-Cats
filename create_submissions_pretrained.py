import logging
import re
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from models.vgg16_pretrained_model import vgg16_pretrained_model

WIDTH = 150
HEIGHT = 150


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert_image_to_data(image):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    array = np.array(image_resized).astype("float32")
    transformed_array = np.concatenate([[array[:, :, 0]], [array[:, :, 1]], [array[:, :, 2]]])
    return transformed_array


def read_data():
    test_data = glob.glob("data/test/*.jpg")
    test_list = [(int(re.split(".jpg", os.path.basename(i))[0]), convert_image_to_data(i)) for i in test_data]
    return test_list


def create_submissions(data):
    prediction = []
    model = vgg16_pretrained_model()
    # data must be 4d (id, depth, width, height) -> np.concatenate([[ data ]])
    for element in data:
       prediction.append((element[0], "{0:.2f}".format(model.predict(np.concatenate([[element[1]]])).tolist()[0][0])))
    return prediction


def write_to_csv(prediction):
    prediction_df = pd.DataFrame(prediction, columns=["id", "label"])
    prediction_df.sort_values(by=["id"], inplace=True)
    prediction_df.to_csv("submissions_pretrained.csv", index=False)


if __name__ == "__main__":
    logger = create_logger()
    logger.info("Read in train data and resize it.")
    test_data = read_data()
    logger.info("Create the submissions for kaggle.")
    prediction = create_submissions(test_data)
    logger.info("Write to csv.")
    write_to_csv(prediction)
