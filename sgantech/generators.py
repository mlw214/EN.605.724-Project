from __future__ import print_function, division

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model, Sequential
import numpy as np

def dense(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))

    noise = Input(shape=input_shape)
    gen = model(noise)

    return Model(noise, gen)

def cnn(input_shape, initial_downsample_res):
    model = Sequential()
    model.add(Dense(np.prod(initial_downsample_res) * 128, activation='relu', input_shape=input_shape))
    model.add(Reshape(initial_downsample_res + (128,)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, activation='tanh', padding='same'))

    noise = Input(shape=input_shape)
    gen = model(noise)

    return Model(noise, gen)
