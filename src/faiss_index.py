import faiss
import numpy as np

# sample embeddings (replace later)
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
]).astype('float32')

# create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# add vectors
index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# sample query
query = np.array([[0.1, 0.2, 0.25]]).astype('float32')

# search
k = 1
distances, indices = index.search(query, k)

print("Nearest index:", indices)
print("Distance:", distances)
