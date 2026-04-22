def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=2):
    sentences = text.split("\n")
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    file_path = "src/data/sample.txt"
    
    text = read_file(file_path)
    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk)
