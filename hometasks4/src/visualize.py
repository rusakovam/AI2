import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

datasetDir = '../dataset/'
embeddingDir = '../embedding/'
embeddingUmapFile = embeddingDir + 'embedding_umap.npy'
visFile = embeddingDir + 'visualize_umap.png'

embedding = np.load(embeddingUmapFile)


def plot():
    # embedding = np.random.random(size=(10, 2)) * 10
    # print(embedding)

    plt.grid(True)
    plt.plot(embedding[:, 0], embedding[:, 1], 'ro')
    plt.show()
    return


def visualize(embedding, vissize=(6000, 6000), scale=0.1, border=500):
    vis = np.zeros(shape=(vissize[0], vissize[1], 3))

    borderWiseSize = np.array(vissize) - 2 * border
    embedding = embedding * (np.divide(borderWiseSize, np.ptp(embedding, axis=0)))

    embedding = embedding - np.min(embedding, axis=0) + border

    images = sorted(os.listdir(datasetDir))

    i = 0
    for f in images:
        print('>', f)
        img = imread(datasetDir + f, mode='RGB')
        imgx = int(img.shape[0] * scale)
        imgy = int(img.shape[1] * scale)
        img = imresize(img, (imgx, imgy, 3))
        x = int(embedding[i, 0] - imgx / 2)
        y = int(embedding[i, 1] - imgy / 2)

        vis[x:x + img.shape[0], y:y + img.shape[1]] = img

        i += 1

    imsave(visFile, vis)

    return

# plot()
visualize(embedding)

print('\nok')