# coding=UTF-8

# Є функція `f(x, y, z)`, де `x, y, z = 0, 1`

# Значення функції

# x y z | f
# 0 0 0 | 1
# 0 0 1 | 0
# 0 1 0 | 1
# 0 1 1 | 0
# 1 0 0 | 1
# 1 0 1 | 1
# 1 1 0 | 1
# 1 1 1 | 1

# напишіть реалізацію нейромережі, яка наближає функцію `f`.

import math

x = [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 1],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 1, 1]]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def g(x):
    return sigmoid(x)

def hTheta(a, x):
    _x = a[0]
    for i in range(0, len(x)):
        _x += a[i+1] * x[i]
    return g(_x)

# a1_AND = [-60, 25, 25, 25]
# a1_OR = [-30, 42, 42, 42]
#
# a2_AND = [-30, 20, 20]
# a2_OR = [-10, 20, 20]


a1 = [-20, -30, 0, 40]
a2 = [10, -20]

def f(x, a1, a2):
    layer1 = hTheta(a1, x)
    layer2 = hTheta(a2, [layer1])
    return layer2


print('a1: ', a1)
print('a2: ', a2)
print('f:')
for i in range(0, len(x)):
    res = f(x[i], a1, a2)
    print(i, '>', '{:06.2f}'.format(res))