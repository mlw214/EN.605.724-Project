from keras.datasets import mnist
import numpy as np


def load_preprocessed_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    return (x_train, y_train), (x_test, y_test)

def learning_curve_wrapper(train_test_fn, x_train, y_train, x_test, y_test, start=None, stop=None, step_size=1000):
    if start is None:
        start = step_size
    if stop is None:
        stop = len(x_train) + step_size
    
    train_losses, test_losses, steps = [], [], []
    for num_examples in np.arange(start, stop, step_size):
        steps.append(num_examples)

        train_loss, test_loss = train_test_fn(num_examples, x_train, y_train, x_test, y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return np.asarray(train_losses), np.asarray(test_losses), np.asarray(steps)
