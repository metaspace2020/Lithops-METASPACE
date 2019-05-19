import logging
from pathlib import Path
import numpy as np
import ibm_boto3
from ibm_botocore.client import Config

logging.getLogger('ibm_boto3').setLevel(logging.CRITICAL)
logging.getLogger('ibm_botocore').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

logger = logging.getLogger('annotation-pipeline')
# handler = logging.StreamHandler()
# format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(format)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)


def get_ibm_cos_client(config):
    return ibm_boto3.client(service_name='s3',
                            ibm_api_key_id=config['ibm_cos']['api_key'],
                            config=Config(signature_version='oauth'),
                            endpoint_url=config['ibm_cos']['endpoint'])


def upload_to_cos(cos_client, src, target_bucket, target_key):
    logger.info('Copying from {} to {}/{}'.format(src, target_bucket, target_key))
    with open(src, "rb") as fp:
        cos_client.put_object(Bucket=target_bucket, Key=target_key, Body=fp)
    logger.info('Copy completed for {}/{}'.format(target_bucket, target_key))


def clean_from_cos(config, bucket, prefix):
    cos_client = get_ibm_cos_client(config)
    objs = cos_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    while 'Contents' in objs:
        keys = [obj['Key'] for obj in objs['Contents']]
        formatted_keys = {'Objects': [{'Key': key} for key in keys]}
        cos_client.delete_objects(Bucket=bucket, Delete=formatted_keys)
        logger.debug(f'Removed {objs["KeyCount"]} objects from {prefix}')
        objs = cos_client.list_objects_v2(Bucket=bucket, Prefix=prefix)


def ds_imzml_path(ds_data_path):
    return next(str(p) for p in Path(ds_data_path).iterdir()
                if str(p).lower().endswith('.imzml'))


def ds_dims(coordinates):
    min_x, min_y = np.amin(coordinates, axis=0)
    max_x, max_y = np.amax(coordinates, axis=0)
    nrows, ncols = max_y - min_y + 1, max_x - min_x + 1
    return nrows, ncols


def get_pixel_indices(coordinates):
    _coord = np.array(coordinates)
    _coord = np.around(_coord, 5)
    _coord -= np.amin(_coord, axis=0)

    _, ncols = ds_dims(coordinates)
    pixel_indices = _coord[:, 1] * ncols + _coord[:, 0]
    pixel_indices = pixel_indices.astype(np.int32)
    return pixel_indices
