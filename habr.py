import os
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pathlib
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, \
    Flatten, Dense, Reshape
from keras.optimizers import Adam
import tensorflow as tf
from typing import *
import time

nb_classes = 34
SIZE = 32

def model():
    model = Sequential()
    model.add(Convolution2D(filters=32,
                            kernel_size=(3, 3),
                            padding='valid',
                            input_shape=(SIZE, SIZE, 1),
                            activation='relu'))
    model.add(Convolution2D(filters=64,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=128,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_classes))

    model.load_weights('weights/comnist_keras_ru.hdf5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def predict_img(model, img):
    letters: str = u'IАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    img_arr = img
    img_arr = img_arr.reshape((1, SIZE, SIZE, 1))

    result = np.argmax(model.predict([img_arr], verbose=0)[0])
    if result == 'I':
        result = u'Ы'
    return letters[result]


def letters_extract(image_file: str, out_size=SIZE):
    gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            letter_crop = cv2.resize(letter_crop, (out_size, out_size))
            letter_square = letter_crop

            # Resize letter to 32х32 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


def img_to_str(model: Any, image_file: str):
    start = time.time()

    letters = letters_extract(image_file)
    print('Количество опознанных букв: ',len(letters))
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] \
            if i < len(letters) - 1 else 0
        s_out += predict_img(model, letters[i][2])
        cv2.imshow('letter',letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '

    print("Время выполнения: %f" % (time.time() - start))
    return s_out


if __name__ == "__main__":
    filepath = "К.png"

    model = model()
    model.save('comnist_letters.h5')
    model = keras.models.load_model('comnist_letters.h5')

    s_out = img_to_str(model, filepath)
    print(s_out)