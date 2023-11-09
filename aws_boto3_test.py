# import creds
import boto3

# session = boto3.Session(
#     aws_access_key_id=creds.ACCESS_KEY, aws_secret_access_key=creds.SECRET_KEY
# )
# s3 = session.resource("s3")
# client = boto3.client("s3")
# response = client.list_buckets()
# print(response)
s3 = boto3.resource("s3")
# print(s3)
# response = s3.list_buckets()
# print(response)
bucket = "nyc-tlc"
# all_objects = s3.list_objects(Bucket=bucket)
# print(all_objects[:3])
count = 0
nyc_bucket = s3.Bucket(bucket)
# for file in nyc_bucket.objects.all():
#     print(file.key)
for obj in nyc_bucket.objects.filter(Prefix="trip data/green_tripdata_"):
    print(obj.key)
    count += 1
    if count == 5:
        break
