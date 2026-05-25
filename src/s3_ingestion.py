import boto3

def read_s3_file():

    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket="sathvik-documind-rag",
        Key="sample.txt"
    )

    text = response["Body"].read().decode("utf-8")
    return text
