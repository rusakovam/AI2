import umap
import os
import numpy as np

vectorsDir = '../vectors/'
embeddingDir = '../embedding/'
embeddingFile = embeddingDir + 'embedding_umap.npy'

vectors = []

vectorFiles = sorted(os.listdir(vectorsDir))
for f in vectorFiles:
    vector = np.load(vectorsDir + f)
    vectors.append(vector)

npVectors = np.array(vectors, dtype=float)

embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(npVectors)

if not os.path.exists(embeddingDir):
    os.makedirs(embeddingDir)

np.save(embeddingFile, embedding)