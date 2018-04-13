# coding=UTF-8

# (Написати функцію тренування для задачі 3)
# Запустити task4_helper_andrew, розібратись, як він працює, і переробити його для мережі з двома прихованими шарами по 21 нейрон, на виході 10 нейронів. Потренувати на MNIST.

import numpy as np
from skimage import io
from mnist import MNIST
import pickle

# dimensions:
_n = 28
n = _n * _n
n1 = 21
n2 = 21
n3 = 10


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    # https://stackoverflow.com/questions/26218617/runtimewarning-overflow-encountered-in-exp-in-computing-the-logistic-function
    return .5 * (1 + np.tanh(.5 * x))


def g(x):
    return sigmoid(x)


# sigmoid derivative:
def dsigmoid(t):
    return t * (1 - t)


def random_parameters(n, n1, n2, n3):
    W1 = 2 * np.random.random(size=(n1, n)) - 1
    b1 = 2 * np.random.random(size=(n1, 1)) - 1
    W2 = 2 * np.random.random(size=(n2, n1)) - 1
    b2 = 2 * np.random.random(size=(n2, 1)) - 1
    W3 = 2 * np.random.random(size=(n3, n2)) - 1
    b3 = 2 * np.random.random(size=(n3, 1)) - 1
    return W1, b1, W2, b2, W3, b3


def print_parameters(W1, b1, W2, b2, W3, b3):
    print('-- Weights and biases: --')
    print('W1:')
    print(W1)
    print('b1:')
    print(b1)
    print('W2:')
    print(W2)
    print('b2:')
    print(b2)
    print('W3:')
    print(W3)
    print('b3:')
    print(b3)
    return


def print_layers(a0, a1, a2, a3):
    print('-- Layers: --')
    print('a0:')
    print(a0)
    print('a1:')
    print(a1)
    print('a2:')
    print(a2)
    print('a3:')
    print(a3)
    return


def nn(x, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(W1, x) + b1
    a1 = g(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = g(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = g(z3)
    return a1, a2, a3, z1, z2, z3


def ys(label):
    y = np.zeros(shape=(n3, 1))
    y[label, 0] = 1
    return y


def loss(a3, y):
    return np.sum(np.power(a3 - y, 2))

def train_gd():
    print('train_gd')
    mndata = MNIST('MNIST')
    images, labels = mndata.load_training()
    print('loaded training data')
    dataset_size = len(images)
    W1, b1, W2, b2, W3, b3 = random_parameters(n, n1, n2, n3)
    alpha = 0.0001
    count = 1000

    for i in range(count):
        losses_k = np.zeros(shape=(dataset_size, 1))
        dC_dW3_ks = np.zeros(shape=(dataset_size, n3, n2))
        dC_db3_ks = np.zeros(shape=(dataset_size, n3, 1))
        dC_dW2_ks = np.zeros(shape=(dataset_size, n2, n1))
        dC_db2_ks = np.zeros(shape=(dataset_size, n2, 1))
        dC_dW1_ks = np.zeros(shape=(dataset_size, n1, n))
        dC_db1_ks = np.zeros(shape=(dataset_size, n1, 1))

        print('--- iteration {:.0f} ---'.format(i))
        for k in range(dataset_size):
            a0 = np.array(images[k]).reshape(n, 1)
            y = ys(labels[k])
            a1, a2, a3, z1, z2, z3 = nn(a0, W1, b1, W2, b2, W3, b3)

            loss_k = loss(a3, y)
            losses_k[k] = loss_k

            error_a3 = a3 - y
            delta_a3 = error_a3 * dsigmoid(z3)
            error_a2 = delta_a3.T.dot(W3).T
            delta_a2 = error_a2 * dsigmoid(z2)
            error_a1 = delta_a2.T.dot(W2).T
            delta_a1 = error_a1 * dsigmoid(z1)

            dC_dW3_ks[k] = delta_a3.dot(a2.T)
            dC_db3_ks[k] = delta_a3
            dC_dW2_ks[k] = delta_a2.dot(a1.T)
            dC_db2_ks[k] = delta_a2
            dC_dW1_ks[k] = delta_a1.dot(a0.T)
            dC_db1_ks[k] = delta_a1

            if (k % 10000) == 0:
                print('t.example {:5.0f}\'s loss: {:.10f}'.format(k, loss_k))

        full_loss = np.mean(np.abs(losses_k))

        dC_dW3 = np.mean(dC_dW3_ks, axis=0)
        dC_db3 = np.mean(dC_db3_ks, axis=0)
        dC_dW2 = np.mean(dC_dW2_ks, axis=0)
        dC_db2 = np.mean(dC_db2_ks, axis=0)
        dC_dW1 = np.mean(dC_dW1_ks, axis=0)
        dC_db1 = np.mean(dC_db1_ks, axis=0)

        W1 -= alpha * dC_dW1
        b1 -= alpha * dC_db1
        W2 -= alpha * dC_dW2
        b2 -= alpha * dC_db2
        W3 -= alpha * dC_dW3
        b3 -= alpha * dC_db3

        # if (j % 10) == 0:
        save_parameters(W1, b1, W2, b2, W3, b3, i, full_loss)
        print('full loss: {:06.06f}'.format(full_loss))

    return


def save_parameters(W1, b1, W2, b2, W3, b3, i, full_loss):
    with open('task4.pkl', 'w') as f:
        pickle.dump([W1, b1, W2, b2, W3, b3, i, full_loss], f)
    return


def load_parameters():
    with open('task4.pkl') as f:
        W1, b1, W2, b2, W3, b3, i, full_loss = pickle.load(f)
    return W1, b1, W2, b2, W3, b3, i, full_loss


def test_nn_random():
    # tests if nn matrix operations work
    print('test_nn_random')
    x = np.random.random(size=(4, 1))
    W1, b1, W2, b2, W3, b3 = random_parameters(4, 5, 5, 10)
    print_parameters(W1, b1, W2, b2, W3, b3)
    a1, a2, a3, z1, z2, z3 = nn(x, W1, b1, W2, b2, W3, b3)
    print_layers(x, a1, a2, a3)
    assert np.shape(a1) == (5, 1)
    assert np.shape(a2) == (5, 1)
    assert np.shape(a3) == (10, 1)
    return


def detect_digit(x, W1, b1, W2, b2, W3, b3):
    a1, a2, output, z1, z2, z3 = nn(x, W1, b1, W2, b2, W3, b3)
    print(output)
    digit = np.argmax(output)
    return digit


def detect_digit_from_file():
    W1, b1, W2, b2, W3, b3, i, full_loss = load_parameters()
    img = io.imread('input.png', as_grey=True)
    x = np.array(img).ravel().reshape(n, 1)
    digit = detect_digit(x, W1, b1, W2, b2, W3, b3)
    print('It\' {:}'.format(digit))
    return


# test_nn_random()
train_gd()
# detect_digit_from_file()
