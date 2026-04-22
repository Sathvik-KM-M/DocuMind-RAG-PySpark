from ingestion import read_file, chunk_text
from embedding import get_embeddings
import faiss
import numpy as np

# 1. Load and chunk document
file_path = "src/data/sample.txt"
text = read_file(file_path)
chunks = chunk_text(text)

print("Chunks created:", len(chunks))

# 2. Convert chunks to embeddings
embeddings = get_embeddings(chunks)
embeddings = np.array(embeddings).astype('float32')

# 3. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index ready with", index.ntotal, "vectors")

# 4. Query
query = input("\nEnter your question: ")

query_embedding = get_embeddings([query])
query_embedding = np.array(query_embedding).astype('float32')

# 5. Search
k = 1
distances, indices = index.search(query_embedding, k)

# 6. Retrieve result
result = chunks[indices[0][0]]

print("\nMost relevant chunk:")
print(result)
