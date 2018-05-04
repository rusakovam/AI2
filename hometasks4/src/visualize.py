import os
import numpy as np
from scipy.misc import imread, imsave, imresize

datasetDir = '../dataset/'
embeddingDir = '../embedding/'
embeddingFile = embeddingDir + 'embedding_tsne.npy'
visFile = embeddingDir + 'visualize_tsne.png'

embedding = np.load(embeddingFile)


def visualize(embedding, visFile, canvasSize=(6000, 6000), iconSize=32, border=500):
    vis = np.zeros(shape=(canvasSize[0], canvasSize[1], 3))

    borderWiseSize = np.array(canvasSize) - 2 * border
    embedding = embedding * (np.divide(borderWiseSize, np.ptp(embedding, axis=0)))

    embedding = embedding - np.min(embedding, axis=0) + border

    images = sorted(os.listdir(datasetDir))

    i = 0
    for f in images:
        print('>', f)
        img = imread(datasetDir + f, mode='RGB')
        ar = img.shape[0] / img.shape[1]
        if ar > 1:
            imgx = int(iconSize)
            imgy = int(iconSize / ar)
        else:
            imgy = int(iconSize)
            imgx = int(iconSize * ar)

        img = imresize(img, (imgx, imgy, 3))
        x = int(embedding[i, 0] - imgx / 2)
        y = int(embedding[i, 1] - imgy / 2)

        vis[x:x + img.shape[0], y:y + img.shape[1]] = img

        i += 1

    imsave(visFile, vis)

    return


visualize(embedding, visFile)

print('\nok')