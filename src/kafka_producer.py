from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

with open("data/sample.txt", "r") as f:
    text = f.read()

producer.send("document-stream", {"document": text})
producer.flush()
