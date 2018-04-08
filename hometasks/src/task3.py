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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def g(x):
    return sigmoid(x)


# dimensions:
_n = 28
n = _n * _n
n1 = 21
n2 = 21
n3 = 10


def detect_digit(x, W1, b1, W2, b2, W3, b3):
    assert len(x) == n, "Input image should be %r x %r pixels, png" % (_n, _n)
    assert np.shape(W1) == (n1, n)
    assert len(b1) == n1
    assert np.shape(W2) == (n2, n1)
    assert len(b2) == n2
    assert np.shape(W3) == (n3, n2)
    assert len(b3) == n3

    a1 = g(np.add(W1.dot(x), b1))
    print('a1:')
    print(a1)
    a2 = g(np.add(W2.dot(a1), b2))
    print('a2:')
    print(a2)
    output = g(np.add(W3.dot(a2), b3))
    print('output:')
    print(output)

    digit = np.argmax(output)
    return digit


img = io.imread('input.png', as_grey=True)
x = np.array(img).ravel()

W1 = np.random.randint(10, size=(n1, n))
b1 = np.random.randint(10, size=n1)
W2 = np.random.randint(10, size=(n2, n1))
b2 = np.random.randint(10, size=n2)
W3 = np.random.randint(10, size=(n3, n2))
b3 = np.random.randint(10, size=n3)

print('x:')
print(x)
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

print('Running:')
d = detect_digit(x, W1, b1, W2, b2, W3, b3)
print('result:')
print(d)