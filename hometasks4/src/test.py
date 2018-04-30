import numpy as np
import os
from scipy.misc import imread, imresize
from keras.applications.resnet50 import ResNet50

file_names = ['dataset/' + s for s in os.listdir('dataset/')]

model = ResNet50(include_top=False, pooling='avg')

img = imread(file_names[0], mode='RGB')
img = imresize(img, (224, 224, 3))


x_batch = np.zeros((1, 224, 224, 3), dtype='float')
x_batch[0] = img

print(x_batch.shape)

prediction = model.predict(x_batch)

np.save('vectors/1.npy', prediction[0])