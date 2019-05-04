import ibm_boto3
import requests
from ibm_botocore.client import Config


def get_ibm_cos_client(config):
    return ibm_boto3.client(service_name='s3',
                            ibm_api_key_id=config['ibm_cos']['api_key'],
                            config=Config(signature_version='oauth'),
                            endpoint_url=config['ibm_cos']['endpoint'])


def copy_local(cos_client, src, target_bucket, target_key):
    print('Copying from {} to {}/{}'.format(src, target_bucket, target_key))
    with open(src, "rb") as fp:
        cos_client.put_object(Bucket=target_bucket, Key=target_key, Body=fp)
    print('Copy completed for {}/{}'.format(target_bucket, target_key))


def copy_url(cos_client, url, target_bucket, target_key):
    print('Downloading {}'.format(url))
    file = requests.get(url).json()['data']
    print('Copying to {}/{}'.format(target_bucket, target_key))
    cos_client.put_object(Bucket=target_bucket,
                          Key=target_key,
                          Body=file.encode('utf-8'))
    print('Copy completed for {}/{}'.format(target_bucket, target_key))
