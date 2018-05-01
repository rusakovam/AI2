# coding=UTF-8

# Задача:
# дописати файл get_vectors.py
# вкидати файли по 16/32 штуки в модель
# зберігати відповідні вектори у numpy.array з відповідними назвами зображень у папку `vectors`


import os

import numpy as np
from keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imresize

datasetDir = '../dataset/'
vectorsDir = '../vectors/'

model = ResNet50(include_top=False, pooling='avg')

batch_size = 16

if not os._exists(vectorsDir):
    os.makedirs(vectorsDir)

file_names = os.listdir(datasetDir)
for i in range(0, len(file_names), batch_size):
    batch = file_names[i: i + batch_size]
    x_batch = np.zeros((len(batch), 224, 224, 3), dtype='float')

    for j, fn in enumerate(batch):
        img = imread(datasetDir + fn, mode='RGB')
        img = imresize(img, (224, 224, 3))
        x_batch[j] = img

    x_batch = x_batch / 127.5 - 1

    prediction = model.predict(x_batch)

    for j, fn in enumerate(batch):
        np.save(vectorsDir + fn + '.npy', prediction[j])