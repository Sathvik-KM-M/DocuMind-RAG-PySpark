def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=50, overlap=10):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


if __name__ == "__main__":
    file_path = "src/data/sample.txt"
    
    text = read_file(file_path)
    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk)
