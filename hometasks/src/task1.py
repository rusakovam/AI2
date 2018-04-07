# coding=UTF-8

# Є табличка з числами:

# x y
# 1 3
# 4 8
# 7 7
# 10 13
# 15 17

# побудувати лінійну регресію
# y = ax + b

# Фактично, треба підібрати параметри `a` і `b` так, щоб сума квадратів помилок була найменша:

# \sum_i (y_i - (a x_i + b))^2 -> min

# Перемагає той, у кого буде найменше значення loss функції

import math
import numpy as np
import matplotlib.pyplot as plt

n = 5
x = [1, 4, 7, 10, 15]
y = [3, 8, 7, 13, 17]

def loss(a, b):
    sum = 0
    for i in range(0, n):
        sum += math.pow(y[i] - (a * x[i] + b), 2)
    return sum

def loss_deriv_da(a, b):
    d = 0
    for i in range(0, n):
        d += (-2) * x[i] * (y[i] - (a * x[i] + b))
    return d

def loss_deriv_db(a, b):
    d = 0
    for i in range(0, n):
        d += (-2) * (y[i] - (a * x[i] + b))
    return d

def plot(a, b):
    plt.axis([0, 20, 0, 20])
    plt.grid(True)
    plt.plot(x, y, 'ro')
    _x = np.linspace(0, 20, 100)
    plt.plot(_x, a * _x + b)
    plt.show()
    return

def graddesc(a0, b0, alpha, count):
    a = a0
    b = b0

    for i in range(0, count):
        l = loss(a, b)
        print(i, '> a: ', a, ', b: ', b, ', loss: ', l)
        tempa = a - alpha * loss_deriv_da(a, b)
        tempb = b - alpha * loss_deriv_db(a, b)
        a = tempa
        b = tempb

    return (a, b)

(a, b) = graddesc(0, 0, 0.001, 1000)
plot(a, b)