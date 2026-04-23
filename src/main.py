from ingestion import chunk_text
from embedding import get_embeddings
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyspark_ingestion import read_documents

# load generator
# generator = pipeline("text-generation", model="google/flan-t5-small")

text = read_documents("src/data/sample.txt")


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 1. Load and chunk document
# file_path = "src/data/sample.txt"
# text = read_file(file_path)
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

print("\nRetrieved context:")
print(result)

prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{result}

Question:
{query}

Answer:
"""

answer = generate_answer(prompt)

print("\nGenerated Answer:")
print(answer)

# print("\nGenerated Answer:")
# print(response[0]['generated_text'])
