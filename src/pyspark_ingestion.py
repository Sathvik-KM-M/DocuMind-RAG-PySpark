from pyspark.sql import SparkSession

def read_documents(file_path):
    spark = SparkSession.builder \
        .appName("DocuMind-RAG") \
        .getOrCreate()

    df = spark.read.text(file_path)
    text = "\n".join([row.value for row in df.collect()])

    # Combine all rows into single text
    #  text = " ".join([row.value for row in df.collect()])
   

    return text


if __name__ == "__main__":
    text = read_documents("data/sample.txt")
    print("Loaded text:")
    print(text)
