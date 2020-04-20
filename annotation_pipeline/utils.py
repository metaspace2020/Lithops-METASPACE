import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import numpy as np
import ibm_boto3
import ibm_botocore
import pandas as pd
import csv

import requests

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
    client_config = ibm_botocore.client.Config(connect_timeout=1,
                                               read_timeout=3,
                                               retries={'max_attempts': 5})

    return ibm_boto3.client(service_name='s3',
                            aws_access_key_id=config['ibm_cos']['access_key'],
                            aws_secret_access_key=config['ibm_cos']['secret_key'],
                            endpoint_url=config['ibm_cos']['endpoint'],
                            config=client_config)


def upload_to_cos(cos_client, src, target_bucket, target_key):
    logger.info('Copying from {} to {}/{}'.format(src, target_bucket, target_key))
    with open(src, "rb") as fp:
        cos_client.put_object(Bucket=target_bucket, Key=target_key, Body=fp)
    logger.info('Copy completed for {}/{}'.format(target_bucket, target_key))


def list_keys(bucket, prefix, cos_client):
    paginator = cos_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    key_list = []
    for page in page_iterator:
        if 'Contents' in page:
            for item in page['Contents']:
                key_list.append(item['Key'])

    logger.info(f'Listed {len(key_list)} objects from {prefix}')
    return key_list


def clean_from_cos(config, bucket, prefix, cos_client=None):
    if not cos_client:
        cos_client = get_ibm_cos_client(config)

    objs = cos_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    removed_keys_n = 0
    while 'Contents' in objs:
        keys = [obj['Key'] for obj in objs['Contents']]
        formatted_keys = {'Objects': [{'Key': key} for key in keys]}
        cos_client.delete_objects(Bucket=bucket, Delete=formatted_keys)
        removed_keys_n += objs["KeyCount"]
        objs = cos_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    logger.info(f'Removed {removed_keys_n} objects from {prefix}')


def ds_imzml_path(ds_data_path):
    return next(str(p) for p in Path(ds_data_path).iterdir()
                if str(p).lower().endswith('.imzml'))


def ds_dims(coordinates):
    min_x, min_y = np.amin(coordinates, axis=0)[:2]
    max_x, max_y = np.amax(coordinates, axis=0)[:2]
    nrows, ncols = max_y - min_y + 1, max_x - min_x + 1
    return nrows, ncols


def get_pixel_indices(coordinates):
    _coord = np.array(coordinates)[:, :2]
    _coord = np.around(_coord, 5)
    _coord -= np.amin(_coord, axis=0)

    _, ncols = ds_dims(coordinates)
    pixel_indices = _coord[:, 1] * ncols + _coord[:, 0]
    pixel_indices = pixel_indices.astype(np.int32)
    return pixel_indices


def append_pywren_stats(futures, memory_mb, plus_objects=0, minus_objects=0):
    if type(futures) != list:
        futures = [futures]

    def calc_cost(runtimes, memory_gb):
        unit_price_in_dollars = 0.000017
        return sum([unit_price_in_dollars * memory_gb * runtime for runtime in runtimes])

    actions_num = len(futures)
    func_name = futures[0].function_name
    runtimes = [future._call_status['exec_time'] for future in futures]
    cost = calc_cost(runtimes, memory_mb / 1024)
    headers = ['Function', 'Actions', 'Memory', 'AvgRuntime', 'Cost', '+Objects', '-Objects']
    content = [func_name, actions_num, memory_mb, np.average(runtimes), cost, plus_objects, minus_objects]

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
    print('Total PyWren cost: {:.3f} $'.format(stats['Cost'].sum()))
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


def read_ranges_from_url(url, ranges):
    """
    Download partial ranges from a file over HTTP. This combines adjacent/overlapping ranges
    to minimize the number of HTTP requests without wasting any bandwidth if there are large gaps
    between requested ranges.
    """
    MAX_JUMP = 2 ** 16 # Largest gap between ranges before a new request should be made

    request_ranges = []
    tasks = []
    range_start = None
    range_end = None
    for input_i in np.argsort(np.array(ranges)[:, 0]):
        lo, hi = ranges[input_i]
        if range_start is None:
            range_start, range_end = lo, hi
        elif lo - range_end <= MAX_JUMP:
            range_end = max(range_end, hi)
        else:
            request_ranges.append((range_start, range_end))
            range_start, range_end = lo, hi

        tasks.append((input_i, len(request_ranges), lo - range_start, hi - range_start))

    if range_start is not None:
        request_ranges.append((range_start, range_end))

    print(f'Reading {len(request_ranges)} ranges: {request_ranges}')

    with ThreadPoolExecutor() as ex:
        def get_range(lo_hi):
            lo, hi = lo_hi
            return requests.get(url, headers={'Range': f'bytes={lo}-{hi}'}).content
        request_results = list(ex.map(get_range, request_ranges))

    return [request_results[request_i][request_lo:request_hi]
            for input_i, request_i, request_lo, request_hi in sorted(tasks)]
