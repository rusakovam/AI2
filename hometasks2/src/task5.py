# coding=UTF-8

# Try MNIST on Keras with hidden layers 21, 21, mse, sgd, and softmax as the activation function.

import numpy as np
from skimage import io
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist


loss = 'mse'
optimizer='sgd'
epochs=500


def prepare_x(x):
    size = len(x)
    x = np.reshape(x, size * 784)
    x = np.reshape(x, (size, 784))
    # print('reshaped x: ', np.shape(x))
    return x


def prepare_y(labels):
    size = len(labels)
    y = np.zeros(shape=(size, 10))
    for i in range(size):
        y[i] = keras.utils.to_categorical(labels[i], num_classes=10)
    # print('y is one-hot now: ', np.shape(y), '- label ', labels[0], ' is ', y[0])
    return y


model = Sequential([
    Dense(21, input_shape=(784,)), Activation('sigmoid'),
    Dense(21,), Activation('sigmoid'),
    Dense(10), Activation('softmax'),
])

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = prepare_x(x_train)
y_train = prepare_y(y_train)
x_test = prepare_x(x_test)
y_test = prepare_y(y_test)


def train():
    print('Training model with MNIST data')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    file = 'weights-mnist-{}-{}-{}.h5py'.format(loss, optimizer, epochs)
    model.save_weights(file)
    return


def load_weights(file):
    print('Loading weights: ', file)
    model.load_weights(file, by_name=False)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print('score: ', score)
    return


def label_from_classifier(y):
    return np.argmax(y)


def predict_input(input, weights):
    load_weights(weights)

    print('Predicting input from file: ', input)
    img = io.imread(input, as_grey=True)
    x = np.array(img).ravel()
    x = np.reshape(x, (1, 784))
    p = model.predict(x, verbose=1)

    print('Prediction is: ', label_from_classifier(p))
    return


def predict_test(i, weights):
    load_weights(weights)

    print('Predicting test example ', i)
    x = x_test[i]
    x = np.reshape(x, (1,784))
    p = model.predict(x, verbose=1)

    print('Prediction is: ', label_from_classifier(p))
    print('Expected value is: ', np.argmax(y_test[i]))
    return


def predict_batch(weights):
    load_weights(weights)
    print('Running batch prediction')
    model.load_weights(weights, by_name=False)
    p = model.predict_on_batch(x_test)
    size = len(p)
    for i in range(size):
        actual = label_from_classifier(p[i])
        expected = label_from_classifier(y_test[i])
        print('test {}. actual: {}, expected: {}'.format(i, actual, expected))
    return


def export_mnist():
    if not os.path.exists('mnist'):
        os.makedirs('mnist')
    size = len(x_test)
    for i in range(size):
        img = x_test[i]
        img = img.reshape((28, 28))
        y = label_from_classifier(y_test[i])
        io.imsave('mnist/mnist_{}_{}.png'.format(i, y), img)
    return


# export_mnist()
# train()
predict_input('test_1.png', 'weights-mnist-sgd-mse-1000.h5py')
# predict_test(17, 'weights-mnist-sgd-mse-1000.h5py')
# predict_batch('weights-mnist-sgd-mse-1000.h5py')


