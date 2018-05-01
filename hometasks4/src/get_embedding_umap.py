import umap
import os
import numpy as np

vectorsDir = '../vectors/'
embeddingDir = '../embedding/'
embeddingFile = embeddingDir + 'embedding_umap.npy'

vectors = []

vectorFiles = os.listdir(vectorsDir)
for f in vectorFiles:
    vector = np.load(vectorsDir + f)
    vectors.append(vector)

npVectors = np.array(vectors, dtype=float)

embedding = umap.UMAP().fit_transform(npVectors)

if not os._exists(embeddingDir):
    os.makedirs(embeddingDir)

np.save(embeddingFile, embedding)