import urllib.request
from pyimzml.ImzMLParser import ImzMLParser
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from time import time
import pandas as pd
from shutil import rmtree

IMZML_URL = 'https://s3.eu-de.cloud-object-storage.appdomain.cloud/metaspace-public/metabolomics/ds/Brain02_Bregma1-42_02/Brain02_Bregma1-42_02.imzML'
IBD_URL = 'https://s3.eu-de.cloud-object-storage.appdomain.cloud/metaspace-public/metabolomics/ds/Brain02_Bregma1-42_02/Brain02_Bregma1-42_02.ibd'
DATASET_PATH = 'ds'
SEGMENTS_PATH = 'segms'
SEGMENT_SIZE_MB = 5
SEGMENTS_BUCKET_NAME = 'omeruseast'
SEGMENTS_COS_PREFIX = 'metabolomics/vm_segments'

from pywren_ibm_cloud.storage import InternalStorage
from pywren_ibm_cloud.config import default_config, extract_storage_config

PYWREN_CONFIG = default_config()
STORAGE_CONFIG = extract_storage_config(PYWREN_CONFIG)
STORAGE = InternalStorage(STORAGE_CONFIG).storage_handler


def download_dataset(imzml_url, idb_url, local_path):
    Path(local_path).mkdir(exist_ok=True)
    imzml_path = f'{local_path}/ds.imzML'
    ibd_path = f'{local_path}/ds.ibd'

    print('Downloading dataset imzML...', end=' ', flush=True)
    t = time()
    urllib.request.urlretrieve(imzml_url, imzml_path)
    print('DONE {:.2f} sec'.format(time() - t))
    print(' * imzML size: {:.2f} mb'.format(Path(imzml_path).stat().st_size // (1024 ** 2)))

    print('Downloading dataset ibd...', end=' ', flush=True)
    t = time()
    urllib.request.urlretrieve(idb_url, ibd_path)
    print('DONE {:.2f} sec'.format(time() - t))
    print(' * ibd size: {:.2f} mb'.format(Path(ibd_path).stat().st_size // (1024 ** 2)))


def ds_imzml_path(ds_data_path):
    return next(str(p) for p in Path(ds_data_path).iterdir()
                if str(p).lower().endswith('.imzml'))


def parse_dataset_chunks(imzml_parser, coordinates, max_size=512 * 1024 ** 2):
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

    sp_id_to_idx = get_pixel_indices(coordinates)
    dtype = imzml_parser.mzPrecision

    def sort_dataset_chunk(sp_inds_list, mzs_list, ints_list):
        mzs = np.concatenate(mzs_list)
        by_mz = np.argsort(mzs)
        sp_mz_int_buf = np.array([np.concatenate(sp_inds_list)[by_mz],
                                  mzs[by_mz],
                                  np.concatenate(ints_list)[by_mz]], dtype).T
        return sp_mz_int_buf

    curr_sp_i = 0
    sp_inds_list, mzs_list, ints_list = [], [], []

    estimated_size_mb = 0
    for x, y in coordinates:
        mzs_, ints_ = imzml_parser.getspectrum(curr_sp_i)
        mzs_, ints_ = map(np.array, [mzs_, ints_])
        sp_idx = sp_id_to_idx[curr_sp_i]
        sp_inds_list.append(np.ones_like(mzs_) * sp_idx)
        mzs_list.append(mzs_)
        ints_list.append(ints_)
        estimated_size_mb += 2 * mzs_.nbytes + ints_.nbytes
        curr_sp_i += 1
        if estimated_size_mb > max_size:
            yield sort_dataset_chunk(sp_inds_list, mzs_list, ints_list)
            sp_inds_list, mzs_list, ints_list = [], [], []
            estimated_size_mb = 0

    if len(sp_inds_list) > 0:
        yield sort_dataset_chunk(sp_inds_list, mzs_list, ints_list)


def define_ds_segments(imzml_parser, sp_n, ds_segm_size_mb=5, sample_sp_n=1000):
    print('Defining segments bounds...', end=' ', flush=True)
    t = time()

    sample_ratio = sample_sp_n / sp_n

    def spectra_sample_gen(imzml_parser, sample_ratio):
        sp_n = len(imzml_parser.coordinates)
        sample_size = int(sp_n * sample_ratio)
        sample_sp_inds = np.random.choice(np.arange(sp_n), sample_size)
        for sp_idx in sample_sp_inds:
            mzs, ints = imzml_parser.getspectrum(sp_idx)
            yield sp_idx, mzs, ints

    spectra_sample = list(spectra_sample_gen(imzml_parser, sample_ratio=sample_ratio))

    spectra_mzs = np.array([mz for sp_id, mzs, ints in spectra_sample for mz in mzs])
    total_n_mz = spectra_mzs.shape[0] / sample_ratio

    float_prec = 4 if imzml_parser.mzPrecision == 'f' else 8
    segm_arr_columns = 3
    segm_n = segm_arr_columns * (total_n_mz * float_prec) // (ds_segm_size_mb * 2 ** 20)
    segm_n = max(1, int(segm_n))

    segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n + 1)]
    segm_lower_bounds = [np.quantile(spectra_mzs, q) for q in segm_bounds_q]
    ds_segments_bounds = np.array(list(zip(segm_lower_bounds[:-1], segm_lower_bounds[1:])))

    max_mz_value = 10 ** 5
    ds_segments_bounds[0, 0] = 0
    ds_segments_bounds[-1, 1] = max_mz_value

    print('DONE {:.2f} sec'.format(time() - t))
    print(' * segments number:', segm_n)
    return ds_segments_bounds


def segment_spectra_chunk(sp_mz_int_buf, ds_segments_bounds, ds_segments_path):
    for segm_i, (l, r) in enumerate(ds_segments_bounds):
        segm_start, segm_end = np.searchsorted(sp_mz_int_buf[:, 1], (l, r))
        pd.to_msgpack(f'{ds_segments_path}/ds_segm_{segm_i:04}.msgpack',
                      sp_mz_int_buf[segm_start:segm_end],
                      append=True)


def upload_segments(ds_segments_path, segments_n, segments_bucket_name, segments_prefix, storage):
    print('Uploading segments...', end=' ', flush=True)
    t = time()

    def _upload(segm_i):
        storage.cos_client.upload_file(Filename=f'{ds_segments_path}/ds_segm_{segm_i:04}.msgpack',
                                       Bucket=segments_bucket_name,
                                       Key=f'{segments_prefix}/{segm_i}.msgpack')

    with ThreadPoolExecutor() as pool:
        pool.map(_upload, range(segments_n))

    print('DONE {:.2f} sec'.format(time() - t))


def segment_dataset():
    rmtree(DATASET_PATH, ignore_errors=True)
    Path(DATASET_PATH).mkdir(parents=True)
    download_dataset(IMZML_URL, IBD_URL, DATASET_PATH)

    imzml_parser = ImzMLParser(ds_imzml_path(DATASET_PATH))
    coordinates = [coo[:2] for coo in imzml_parser.coordinates]
    sp_n = len(coordinates)

    ds_segments_bounds = define_ds_segments(imzml_parser, sp_n, ds_segm_size_mb=SEGMENT_SIZE_MB)
    segments_n = len(ds_segments_bounds)

    print('Segmenting...', end=' ', flush=True)
    t = time()
    rmtree(SEGMENTS_PATH, ignore_errors=True)
    Path(SEGMENTS_PATH).mkdir(parents=True)
    for sp_mz_int_buf in parse_dataset_chunks(imzml_parser, coordinates):
        segment_spectra_chunk(sp_mz_int_buf, ds_segments_bounds, SEGMENTS_PATH)
    print('DONE {:.2f} sec'.format(time() - t))

    upload_segments(SEGMENTS_PATH, segments_n, SEGMENTS_BUCKET_NAME, SEGMENTS_COS_PREFIX, STORAGE)


if __name__ == '__main__':
    segment_dataset()
