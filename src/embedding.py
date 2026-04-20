from sentence_transformers import SentenceTransformer

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "The warranty period is 2 years",
        "Returns allowed within 30 days"
    ]

    vectors = get_embeddings(sample_chunks)

    print("Number of vectors:", len(vectors))
    print("Vector for first chunk:", vectors[0])
