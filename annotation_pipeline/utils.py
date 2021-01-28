import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
from io import BytesIO
import pickle
import requests
from lithops.storage.utils import CloudObject, StorageNoSuchKeyError

logging.getLogger('ibm_boto3').setLevel(logging.CRITICAL)
logging.getLogger('ibm_botocore').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('engine').setLevel(logging.CRITICAL)

logger = logging.getLogger('annotation-pipeline')


class PipelineStats:
    path = None

    @classmethod
    def init(cls):
        Path('logs').mkdir(exist_ok=True)
        cls.path = datetime.now().strftime("logs/%Y-%m-%d_%H:%M:%S.csv")
        headers = ['Function', 'Actions', 'Memory', 'AvgRuntime', 'Cost', 'CloudObjects']
        pd.DataFrame([], columns=headers).to_csv(cls.path, index=False)

    @classmethod
    def _append(cls, content):
        pd.DataFrame(content).to_csv(cls.path, mode='a', header=False, index=False)

    @classmethod
    def append_func(cls, futures, memory_mb, cloud_objects_n=0):
        if type(futures) != list:
            futures = [futures]

        def calc_cost(runtimes, memory_gb):
            cost_in_dollars_per_gb_sec = 0.000017
            return sum([cost_in_dollars_per_gb_sec * memory_gb * runtime for runtime in runtimes])

        actions_num = len(futures)
        func_name = futures[0].function_name
        runtimes = [future.stats['worker_exec_time'] for future in futures]
        cost = calc_cost(runtimes, memory_mb / 1024)
        cls._append([[func_name, actions_num, memory_mb, np.average(runtimes), cost, cloud_objects_n]])

    @classmethod
    def append_vm(cls, func_name, exec_time, cloud_objects_n=0):
        cls._append([[func_name, 'VM', '', exec_time, '', cloud_objects_n]])

    @classmethod
    def get(cls):
        stats = pd.read_csv(cls.path)
        print('Total cost: {:.3f} $ (Using IBM Cloud pricing)'.format(stats['Cost'].sum()))
        return stats


def upload_if_needed(storage, src, target_bucket, target_prefix=None):
    example_prefix = 'cos://embl-datasets/'
    if src.startswith(example_prefix):
        can_access_directly = (
            storage.backend in ('ibm_cos', 'cos')
            and object_exists(storage, 'embl-datasets', src[len(example_prefix):])
        )
        if not can_access_directly:
            # If using the sample datasets with a non-COS storage backend, use HTTPS instead
            logger.info(f'Translating IBM COS path to public HTTPS path for example file "{src}"')
            src = src.replace(example_prefix, 'https://s3.us-east.cloud-object-storage.appdomain.cloud/embl-datasets/')

    if '://' in src:
        backend, path = src.split('://', maxsplit=1)
        bucket, key = path.split('/', maxsplit=1)
    else:
        backend = None
        bucket = None
        filename = Path(src).name
        key = f'{target_prefix}/{filename}' if target_prefix else filename

    if backend not in ('https', 'http', None):
        # If it's not HTTP / filesystem, assume it's a bucket/key that Lithops can find
        assert object_exists(storage, bucket, key), f'Could not resolve input path "{src}"'
        return CloudObject(storage.backend, bucket, key)

    if object_exists(storage, target_bucket, key):
        # If the file would have to be uploaded, but there's already a copy in the storage bucket, use it
        logger.debug(f'Found input file already uploaded at "{storage.backend}://{target_bucket}/{key}"')
        return CloudObject(storage.backend, target_bucket, key)
    else:
        # Upload from HTTP or filesystem
        if backend in ('https', 'http'):
            r = requests.get(src, stream=True)
            r.raise_for_status()
            stream = r.raw
        else:
            src_path = Path(src)
            assert src_path.exists(), f'Could not find input file "{src}"'
            stream = src_path.open('rb')

        logger.info(f'Uploading "{src}" to "{storage.backend}://{target_bucket}/{key}"')
        if hasattr(storage.get_client(), 'upload_fileobj'):
            # Try a streaming upload through boto3 interface
            storage.get_client().upload_fileobj(Fileobj=stream, Bucket=target_bucket, Key=key)
            return CloudObject(storage.backend, target_bucket, key)
        else:
            # Fall back to buffering the entire object in memory for other backends
            data = stream.read()
            return storage.put_cloudobject(data, target_bucket, key)


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


def serialise_to_file(obj, path):
    with open(path, 'wb') as file:
        file.write(pa.serialize(obj).to_buffer())


def deserialise_from_file(path):
    with open(path, 'rb') as file:
        data = pa.deserialize(file.read())
    return data


def serialise(obj):
    try:
        return BytesIO(pa.serialize(obj).to_buffer())
    except pa.lib.SerializationCallbackError:
        return BytesIO(pickle.dumps(obj))


def deserialise(data):
    if hasattr(data, 'read'):
        data = data.read()
    try:
        return pa.deserialize(data)
    except pa.lib.ArrowInvalid:
        return pickle.loads(data)


def object_exists(storage, bucket, key):
    try:
        storage.head_object(bucket, key)
        return True
    except StorageNoSuchKeyError:
        return False


def read_object_with_retry(storage, bucket, key, stream_reader=None):
    last_exception = None
    for attempt in range(1, 4):
        try:
            logger.debug(f'Reading {key} (attempt {attempt})')
            data_stream = storage.get_object(bucket, key, stream=True)
            if stream_reader:
                data = stream_reader(data_stream)
            else:
                data = data_stream.read()
            length = getattr(data_stream, '_amount_read', 'Unknown')
            logger.debug(f'Reading {key} (attempt {attempt}) - Success ({length} bytes)')
            return data
        except Exception as ex:
            logger.debug(f'Exception reading {key} (attempt {attempt}): ', ex)
            last_exception = ex
    raise last_exception


def read_cloud_object_with_retry(storage, cobject, stream_reader=None):
    last_exception = None
    for attempt in range(1, 4):
        try:
            logger.debug(f'Reading {cobject.key} (attempt {attempt})')
            data_stream = storage.get_cloudobject(cobject, stream=True)
            if stream_reader:
                data = stream_reader(data_stream)
            else:
                data = data_stream.read()
            length = getattr(data_stream, '_amount_read', 'Unknown')
            logger.debug(f'Reading {cobject.key} (attempt {attempt}) - Success ({length} bytes)')
            return data
        except Exception as ex:
            logger.debug(f'Exception reading {cobject.key} (attempt {attempt}): ', ex)
            last_exception = ex
    raise last_exception


def read_ranges_from_url(storage, cobj_or_url, ranges):
    """
    Download partial ranges from a file over HTTP/COS/Lithops. This combines adjacent/overlapping ranges
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
            headers = {'Range': f'bytes={lo}-{hi}'}
            if isinstance(cobj_or_url, CloudObject):
                return storage.get_object(cobj_or_url.bucket, cobj_or_url.key, extra_get_args=headers)
            else:
                return requests.get(cobj_or_url, headers=headers).content
        request_results = list(ex.map(get_range, request_ranges))

    return [request_results[request_i][request_lo:request_hi]
            for input_i, request_i, request_lo, request_hi in sorted(tasks)]
