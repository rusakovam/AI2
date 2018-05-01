import matplotlib.pyplot as plt
import numpy as np

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


plot()