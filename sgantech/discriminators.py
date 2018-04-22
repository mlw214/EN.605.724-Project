from __future__ import print_function, division

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model, Sequential
import numpy as np

def dense(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    real_fake_input = Input(shape=input_shape)
    label_input = Input(shape=input_shape)

    real_fake_features = model(real_fake_input)
    label_features = model(label_input)
    valid = Dense(1, activation="sigmoid")(real_fake_features)
    label = Dense(num_classes+1, activation="softmax")(label_features)

    return Model([real_fake_input, label_input], [valid, label])

def cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=input_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())

    real_fake_input = Input(shape=input_shape)
    label_input = Input(shape=input_shape)

    real_fake_features = model(real_fake_input)
    label_features = model(label_input)
    valid = Dense(1, activation="sigmoid")(real_fake_features)
    label = Dense(num_classes+1, activation="softmax")(label_features)

    return Model([real_fake_input, label_input], [valid, label])
