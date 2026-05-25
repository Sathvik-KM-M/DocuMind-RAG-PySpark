from ingestion import chunk_text
from embedding import get_embeddings
from s3_ingestion import read_s3_file

import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load document from S3
text = read_s3_file()

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Generate answer
def generate_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Chunking
chunks = chunk_text(text)

print("Chunks created:", len(chunks))

# Embeddings
embeddings = get_embeddings(chunks)
embeddings = np.array(embeddings).astype("float32")

# FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index ready with", index.ntotal, "vectors")

# Ask question
query = input("\nEnter your question: ")

# Query embedding
query_embedding = get_embeddings([query])
query_embedding = np.array(query_embedding).astype("float32")

# Search
k = 1

distances, indices = index.search(query_embedding, k)

# Retrieve result
result = chunks[indices[0][0]]

print("\nRetrieved context:")
print(result)

# Prompt
prompt = f"""
Answer the question ONLY using the context below.

If the answer is not in the context,
say "I don't know".

Context:
{result}

Question:
{query}

Answer:
"""

# Generate final answer
answer = generate_answer(prompt)

print("\nGenerated Answer:")
print(answer)
