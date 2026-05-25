from ingestion import chunk_text
from embedding import get_embeddings
from ingestion import read_file

import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pyspark.sql import SparkSession

# -----------------------------
# STEP 1: CREATE SPARK SESSION
# -----------------------------

spark = SparkSession.builder \
    .appName("KafkaToRAGPipeline") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -----------------------------
# STEP 2: READ DATA FROM KAFKA
# -----------------------------

df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "documents") \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load()

print("Kafka data loaded")

# Convert Kafka binary data to string
messages = df.selectExpr("CAST(value AS STRING)")

# Collect Kafka messages
rows = messages.collect()

print("Kafka messages collected")

# Stop Spark session to free memory
spark.stop()

# -----------------------------
# STEP 3: COMBINE TEXT
# -----------------------------

text = " ".join([row.value for row in rows])

print("\nDocuments received from Kafka:\n")
print(text)

# -----------------------------
# STEP 4: SAVE KAFKA DATA
# -----------------------------

with open("kafka_data.txt", "w") as f:
    f.write(text)

print("\nKafka data saved to kafka_data.txt")

# -----------------------------
# STEP 5: LOAD SAVED TEXT
# -----------------------------

text = read_file("kafka_data.txt")

# -----------------------------
# STEP 6: LOAD MODEL
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

# -----------------------------
# STEP 7: GENERATE ANSWER
# -----------------------------
def generate_answer(prompt):

    print("Tokenizer running...")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    print("Model generation started...")

    outputs = model.generate(
        **inputs,
        max_new_tokens=10
    )

    print("Model generation finished...")

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
# -----------------------------
# STEP 8: CHUNKING
# -----------------------------

chunks = chunk_text(text)

print("\nChunks created:", len(chunks))

# -----------------------------
# STEP 9: EMBEDDINGS
# -----------------------------

embeddings = get_embeddings(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

# -----------------------------
# STEP 10: CREATE FAISS INDEX
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(
    "FAISS index ready with",
    index.ntotal,
    "vectors"
)

# -----------------------------
# STEP 11: USER QUESTION
# -----------------------------

#query = input("\nEnter your question: ")
query = "What is 2+2?"
# -----------------------------
# STEP 12: QUERY EMBEDDING
# -----------------------------

print("Generating query embedding...")

query_embedding = get_embeddings([query])

query_embedding = np.array(
    query_embedding
).astype("float32")

print("Query embedding complete")

# -----------------------------
# STEP 13: SEARCH
# -----------------------------

print("Searching FAISS...")

k = 1

distances, indices = index.search(
    query_embedding,
    k
)

# -----------------------------
# STEP 14: RETRIEVE CONTEXT
# -----------------------------

result = chunks[indices[0][0]]

print("\nRetrieved context:")
print(result)

# -----------------------------
# STEP 15: CREATE PROMPT
# -----------------------------

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

# -----------------------------
# STEP 16: GENERATE FINAL ANSWER
# -----------------------------

print("Generating final answer...")

answer = generate_answer(prompt)

print("Generation complete")

# -----------------------------
# STEP 17: PRINT ANSWER
# -----------------------------

print("\nGenerated Answer:")
print(answer)
