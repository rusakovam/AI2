# coding=UTF-8

# На будь-якій мові написати реалізацію forward propagation для нейромережі, задача якої - розпізнавати рукописні цифри.
# Нейромережа має складатись з двох прихованих шарів розміру 21.
# Формули для обчислення нейромережі є на дошці. Реалізація має виглядати так:

# function detect_digit(x, W1, b1, W2, b2, W3, b3) {
# // x - зображення 28х28
# // magic goes here
# digit = ...; // 0, ..., 9
# return digit;
# }

# Параметри W1, b1, W2, b2, W3, b3 тренувати не потрібно. Можете ініціалізувати їх випадковими значеннями.
# Потрібен тільки робочий код нейромережі та test case.


import numpy as np
from skimage import io
from mnist import MNIST


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    # https://stackoverflow.com/questions/26218617/runtimewarning-overflow-encountered-in-exp-in-computing-the-logistic-function
    return .5 * (1 + np.tanh(.5 * x))


def g(x):
    return sigmoid(x)


# dimensions:
_n = 28
n = _n * _n
n1 = 21
n2 = 21
n3 = 10


def nn(x, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(W1, x) + b1
    a1 = g(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = g(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = g(z3)
    return a1, a2, a3, z1, z2, z3


def detect_digit(x, W1, b1, W2, b2, W3, b3, verbose=False):
    assert len(x) == n, "Input image should be %r x %r pixels, png" % (_n, _n)
    assert np.shape(W1) == (n1, n)
    assert len(b1) == n1
    assert np.shape(W2) == (n2, n1)
    assert len(b2) == n2
    assert np.shape(W3) == (n3, n2)
    assert len(b3) == n3

    a1, a2, output, z1, z2, z3 = nn(x, W1, b1, W2, b2, W3, b3)
    digit = np.argmax(output)

    if verbose:
        print('-- Input (x): --')
        print(np.reshape(x, (_n, _n)))
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
        print('-- Layers: --')
        print('a1:')
        print(a1)
        print('a2:')
        print(a2)
        print('output:')
        print(output)
        print('result:')
        print(digit)
    return digit


def loss(output, y):
    return np.sum(np.power(y - output, 2))


def random_parameters():
    W1 = 2 * np.random.random(size=(n1, n)) - 1
    b1 = 2 * np.random.random(size=n1) - 1
    W2 = 2 * np.random.random(size=(n2, n1)) - 1
    b2 = 2 * np.random.random(size=n2) - 1
    W3 = 2 * np.random.random(size=(n3, n2)) - 1
    b3 = 2 * np.random.random(size=n3) - 1
    return W1, b1, W2, b2, W3, b3


def test_detect_digit():
    W1, b1, W2, b2, W3, b3 = random_parameters()
    images, labels = mndata.load_testing()
    i = 0
    x = images[i]
    d = detect_digit(x, W1, b1, W2, b2, W3, b3, verbose=True)
    print('I think it\'s {:1.0f}'.format(d))
    print('MNIST thinks it\'s {:1.0f}'.format(labels[i]))
    return


# random file:
# file = 'input.png'
# img = io.imread(file, as_grey=True)
# x = np.array(img).ravel()

mndata = MNIST('MNIST')
test_detect_digit()