import pickle
import numpy as np

with open("models/scheme_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(type(embeddings))        # should be <class 'numpy.ndarray'>
print(embeddings.shape)        # e.g., (500, 384)
print(embeddings[0])           # print the first embedding vector
