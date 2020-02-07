import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import ibm_boto3
import pandas as pd
import csv

logging.getLogger('ibm_boto3').setLevel(logging.CRITICAL)
logging.getLogger('ibm_botocore').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

logger = logging.getLogger('annotation-pipeline')

# handler = logging.StreamHandler()
# format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(format)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

STATUS_PATH = datetime.now().strftime("logs/%Y-%m-%d_%H:%M:%S.csv")


def get_ibm_cos_client(config):
    return ibm_boto3.client(service_name='s3',
                            aws_access_key_id=config['ibm_cos']['access_key'],
                            aws_secret_access_key=config['ibm_cos']['secret_key'],
                            endpoint_url=config['ibm_cos']['endpoint'])


def upload_to_cos(cos_client, src, target_bucket, target_key):
    logger.info('Copying from {} to {}/{}'.format(src, target_bucket, target_key))
    with open(src, "rb") as fp:
        cos_client.put_object(Bucket=target_bucket, Key=target_key, Body=fp)
    logger.info('Copy completed for {}/{}'.format(target_bucket, target_key))


def clean_from_cos(config, bucket, prefix, cos_client=None):
    if not cos_client:
        cos_client = get_ibm_cos_client(config)
    objs = cos_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    while 'Contents' in objs:
        keys = [obj['Key'] for obj in objs['Contents']]
        formatted_keys = {'Objects': [{'Key': key} for key in keys]}
        cos_client.delete_objects(Bucket=bucket, Delete=formatted_keys)
        logger.info(f'Removed {objs["KeyCount"]} objects from {prefix}')
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


def append_pywren_stats(futures, memory, plus_objects=0, minus_objects=0):
    if type(futures) != list:
        futures = [futures]

    actions_num = len(futures)
    func_name = futures[0].function_name
    average_runtime = np.average([future._call_status['exec_time'] for future in futures])
    headers = ['Function', 'Actions', 'Memory', 'Runtime', '+Objects', '-Objects']
    content = [func_name, actions_num, memory, average_runtime, plus_objects, minus_objects]

    Path('logs').mkdir(exist_ok=True)
    if not Path(STATUS_PATH).exists():
        with open(STATUS_PATH, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

    with open(STATUS_PATH, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writerow(dict(zip(headers, content)))


def get_pywren_stats(log_path=STATUS_PATH):
    stats = pd.read_csv(log_path)
    unit_price_in_dollars = 0.000017
    calc_func = lambda row: row[1] * (row[2] / 1024) * row[3] * unit_price_in_dollars
    print('Total PyWren cost:', np.sum(np.apply_along_axis(calc_func, 1, stats)), '$')
    return stats


def read_object_with_retry(ibm_cos, bucket, key, stream_reader=None):
    last_exception = None
    for attempt in range(1, 4):
        try:
            print(f'Reading {key} (attempt {attempt})')
            data_stream = ibm_cos.get_object(Bucket=bucket, Key=key)['Body']
            if stream_reader:
                data = stream_reader(data_stream)
            else:
                data = data_stream.read()
            length = getattr(data_stream, '_amount_read', 'Unknown')
            print(f'Reading {key} (attempt {attempt}) - Success ({length} bytes)')
            return data
        except Exception as ex:
            print(f'Exception reading {key} (attempt {attempt}): ', ex)
            last_exception = ex
    raise last_exception
