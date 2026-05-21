from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DocuMind-RAG") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "document-stream") \
    .load()

value_df = df.selectExpr("CAST(value AS STRING)")

query = value_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()

