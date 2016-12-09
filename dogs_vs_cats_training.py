import logging
import glob
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score


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
    image_resized = Image.open(image).resize((100, 100))
    cat_array = np.array(image_resized).T
    return cat_array


def create_train_test_data():
    cat_files = glob.glob("data/train/cat*")
    dog_files = glob.glob("data/train/dog*")

    # Restrict cat and dog files here for testing
    cat_list = [convert_image_to_data(i) for i in cat_files]
    dog_list = [convert_image_to_data(i) for i in dog_files]

    y_cat = np.zeros(len(cat_list))
    y_dog = np.ones(len(dog_list))

    X = np.concatenate([cat_list, dog_list])
    X = np.concatenate([cat_list, dog_list])
    y = np.concatenate([y_cat, y_dog])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


def create_model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def train_model(model, X_data_train, y_target_train):
    model.fit(X_data_train, y_target_train, batch_size=32, nb_epoch=20)
    return model


def evaluate_model(model, X_data_test, y_target_test):
    y_test_predict = model.predict(X_data_test)
    return accuracy_score(y_target_test, y_test_predict)


if __name__ == "__main__":
    logger = create_logger()
    logger.info("Create train and test dataset.")
    X_train, X_test, y_train, y_test = create_train_test_data()
    logger.info("Shape for X_train: " + str(X_train.shape) + " Shape for y_train: " + str(y_train.shape))

    logger.info("Create the model.")
    model = create_model()

    logger.info("Train the model.")
    trained_model = train_model(model, X_train, y_train)

    logger.info("Evaluate the model.")
    accuracy_score = evaluate_model(trained_model, X_test, y_test)
    logger.info("The accurarcy is " + str(accuracy_score))

    logger.info("Save model")
    trained_model.save("dogs_vs_cats_model_VGG-like_convnet.h5")
