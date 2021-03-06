from tsne import bh_sne
import os
import numpy as np

vectorsDir = '../vectors/'
embeddingDir = '../embedding/'
embeddingFile = embeddingDir + 'embedding_tsne.npy'

vectors = []

vectorFiles = sorted(os.listdir(vectorsDir))
for f in vectorFiles:
    vector = np.load(vectorsDir + f)
    vectors.append(vector)

npVectors = np.array(vectors, dtype=float)

embedding = bh_sne(npVectors)

if not os.path.exists(embeddingDir):
    os.makedirs(embeddingDir, exist_ok=True)

np.save(embeddingFile, embedding)