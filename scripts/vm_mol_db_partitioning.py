from time import time
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import rmtree
from concurrent.futures.thread import ThreadPoolExecutor

CENTROIDS_BUCKET = 'omeruseast'
MOL_DB_PREFIX = 'metabolomics/db/centroids_chunks'
CENTROIDS_SEGMENTS_PREFIX = 'metabolomics/vm_db_segments'
CENTR_SEGM_PATH = '/data/metabolomics/db/segms'
DS_SEGMENTS = np.array([[79.99708557, 145.02574158], [145.02574158, 180.09922791], [180.09922791, 205.08337402],
                        [205.08337402, 227.12495422], [227.12495422, 247.00369263], [247.00369263, 266.15087891],
                        [266.15087891, 284.1260376], [284.1260376, 304.17520142], [304.17520142, 321.31515503],
                        [321.31515503, 340.25341797], [340.25341797, 359.31045532], [359.31045532, 379.28167725],
                        [379.28167725, 406.34417725], [406.34417725, 441.4055481], [441.4055481, 499.97909546]])
DS_SEGM_SIZE_MB = 100

from pywren_ibm_cloud.storage import InternalStorage
from pywren_ibm_cloud.config import default_config, extract_storage_config

PYWREN_CONFIG = default_config()
STORAGE_CONFIG = extract_storage_config(PYWREN_CONFIG)
STORAGE = InternalStorage(STORAGE_CONFIG).storage_handler


def download_database(storage, bucket, prefix):
    keys = storage.list_keys(bucket, prefix)

    def _download(key):
        data_stream = storage.get_object(bucket, key, stream=True)
        return pd.read_msgpack(data_stream).sort_values('mz')

    with ThreadPoolExecutor() as pool:
        centroids_df = pd.concat(list(pool.map(_download, keys)))

    return centroids_df


def clip_centroids_df(centroids_df, mz_min, mz_max):
    ds_mz_range_unique_formulas = centroids_df[(mz_min < centroids_df.mz) &
                                               (centroids_df.mz < mz_max)].index.unique()
    centr_df = centroids_df[centroids_df.index.isin(ds_mz_range_unique_formulas)].reset_index().copy()
    return centr_df


def calculate_centroids_segments_n(centr_df, ds_segments, ds_segm_size_mb):
    ds_size_mb = len(ds_segments) * ds_segm_size_mb
    data_per_centr_segm_mb = 50
    peaks_per_centr_segm = 1e4
    centr_segm_n = int(max(ds_size_mb // data_per_centr_segm_mb,
                           centr_df.shape[0] // peaks_per_centr_segm,
                           32))
    return centr_segm_n


def segment_centroids(centr_df, segm_n, centr_segm_path):
    first_peak_df = centr_df[centr_df.peak_i == 0].copy()
    segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n)]
    segm_lower_bounds = list(np.quantile(first_peak_df.mz, q) for q in segm_bounds_q)

    segment_mapping = np.searchsorted(segm_lower_bounds, first_peak_df.mz.values, side='right') - 1
    first_peak_df['segm_i'] = segment_mapping

    centr_segm_df = pd.merge(centr_df, first_peak_df[['formula_i', 'segm_i']],
                             on='formula_i').sort_values('mz')
    for segm_i, df in centr_segm_df.groupby('segm_i'):
        pd.to_msgpack(f'{centr_segm_path}/centr_segm_{segm_i:04}.msgpack', df)


def upload_segments(centr_segments_path, segments_n, segments_bucket_name, segments_prefix, storage):
    def _upload(segm_i):
        storage.cos_client.upload_file(Filename=f'{centr_segments_path}/centr_segm_{segm_i:04}.msgpack',
                                       Bucket=segments_bucket_name,
                                       Key=f'{segments_prefix}/{segm_i}.msgpack')

    with ThreadPoolExecutor() as pool:
        pool.map(_upload, range(segments_n))


if __name__ == '__main__':
    start = time()

    print('Downloading molecular database...', end=' ', flush=True)
    t = time()
    centroids_df = download_database(STORAGE, CENTROIDS_BUCKET, MOL_DB_PREFIX)
    print('DONE {:.2f} sec'.format(time() - t))
    print(' * centorids:', centroids_df.shape[0])
    centroids_df = centroids_df[centroids_df.mz > 0]

    mz_min, mz_max = DS_SEGMENTS[0, 0], DS_SEGMENTS[-1, 1]
    print('Clipping unrelevant centroids...', end=' ', flush=True)
    t = time()
    centr_df = clip_centroids_df(centroids_df, mz_min, mz_max)
    print('DONE {:.2f} sec'.format(time() - t))
    print(' * centorids:', centr_df.shape[0])

    print('Defining segments number...', end=' ', flush=True)
    t = time()
    centr_segm_n = calculate_centroids_segments_n(centr_df, DS_SEGMENTS, DS_SEGM_SIZE_MB)
    print('DONE {:.2f} sec'.format(time() - t))
    print(' * segments number:', centr_segm_n)

    rmtree(CENTR_SEGM_PATH, ignore_errors=True)
    Path(CENTR_SEGM_PATH).mkdir(parents=True)
    print('Segmenting...', end=' ', flush=True)
    t = time()
    segment_centroids(centr_df, centr_segm_n, CENTR_SEGM_PATH)
    print('DONE {:.2f} sec'.format(time() - t))

    print('Uploading segments...', end=' ', flush=True)
    t = time()
    upload_segments(CENTR_SEGM_PATH, centr_segm_n, CENTROIDS_BUCKET, CENTROIDS_SEGMENTS_PREFIX, STORAGE)
    print('DONE {:.2f} sec'.format(time() - t))

    print('--- {:.2f} sec ---'.format(time() - start))
