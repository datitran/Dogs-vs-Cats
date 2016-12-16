import logging
import glob
import numpy as np

np.random.seed(1337)
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from modified_vgg_16_model import modified_vgg_16l
from google_net import modified_googlenet

"""
64 * 64 for VGG16
224 * 224 for GoogleNet
"""
WIDTH = 224
HEIGHT = 224


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert_image_to_data(image, WIDTH, HEIGHT):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    image_array = np.array(image_resized).T
    return image_array


def create_train_test_data(WIDTH, HEIGHT):
    cat_files = glob.glob("data/train/cat*")
    dog_files = glob.glob("data/train/dog*")

    # Restrict cat and dog files here for testing
    cat_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat_files]
    dog_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog_files]

    y_cat = np.zeros(len(cat_list))
    y_dog = np.ones(len(dog_list))

    X = np.concatenate([cat_list, dog_list])
    X = np.concatenate([cat_list, dog_list])
    y = np.concatenate([y_cat, y_dog])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


def train_model(model, X_data_train, y_target_train, flag):
    """
    :param model: compiled model
    :param X_data_train: 3d array
    :param y_target_train: 1d array
    :param flag: true if googlnet (output expect 3d array) else false if 1d array for output
    :return: fitted model
    """
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    if flag:
        model.fit(X_data_train, [y_target_train] * 3, batch_size=32, nb_epoch=20, validation_split=0.2,
                  callbacks=[early_stopping])
    else:
        model.fit(X_data_train, y_target_train, batch_size=32, nb_epoch=20, validation_split=0.2,
                  callbacks=[early_stopping])
    return model


def evaluate_model(model, X_data_test, y_target_test):
    y_test_predict = model.predict_classes(X_data_test)
    return accuracy_score(y_target_test, y_test_predict)


if __name__ == "__main__":
    logger = create_logger()
    logger.info("Create train and test dataset.")
    X_train, X_test, y_train, y_test = create_train_test_data(WIDTH, HEIGHT)
    logger.info("Shape for X_train: " + str(X_train.shape) + " Shape for y_train: " + str(y_train.shape))

    logger.info("Create the model.")
    # model = modified_vgg_16l(WIDTH, HEIGHT)
    model = modified_googlenet(WIDTH, HEIGHT)

    logger.info("Train the model.")
    # flag=False if no GoogleNet
    trained_model = train_model(model, X_train, y_train, flag=True)

    logger.info("Evaluate the model.")
    accuracy_score = evaluate_model(trained_model, X_test, y_test)
    logger.info("The accurarcy is " + str(accuracy_score))

    logger.info("Save model")
    #trained_model.save("dogs_vs_cats_model_VGG-like_convnet.h5")
    trained_model.save("dogs_vs_cats_model_googlenet.h5")
