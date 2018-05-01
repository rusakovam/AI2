from tsne import bh_sne
import os

import numpy as np

vector_files = os.listdir('vectors/')

vectors = []

for fn in vector_files:
    vector = np.load('vectors/' + fn)
    vectors.append(vector)

np_vectors = np.array(vectors, dtype=float)

vectors_2d = bh_sne(np_vectors)

print('a')