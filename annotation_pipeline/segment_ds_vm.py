from tempfile import TemporaryDirectory
import requests
from pyimzml.ImzMLParser import ImzMLParser
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from time import time
import pandas as pd

from annotation_pipeline.utils import logger
from pywren_ibm_cloud.storage import InternalStorage
from pywren_ibm_cloud.config import default_config, extract_storage_config

PYWREN_CONFIG = default_config()
STORAGE_CONFIG = extract_storage_config(PYWREN_CONFIG)
STORAGE = InternalStorage(STORAGE_CONFIG).storage_handler


def download_dataset(imzml_url, ibd_url, local_path, storage):
    def _download(url, path):
        if url.startswith('cos://'):
            bucket, key = url[len('cos://'):].split('/', maxsplit=1)
            print((bucket, key))
            res = storage.download_file(bucket, key, str(path))
            print(res)
        else:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with path.open('wb') as f:
                    for chunk in r.iter_content():
                        f.write(chunk)

    Path(local_path).mkdir(exist_ok=True)
    imzml_path = local_path / 'ds.imzML'
    ibd_path = local_path / 'ds.ibd'

    with ThreadPoolExecutor() as ex:
        ex.map(_download, [imzml_url, ibd_url], [imzml_path, ibd_path])

    imzml_size = imzml_path.stat().st_size / (1024 ** 2)
    ibd_size = ibd_path.stat().st_size / (1024 ** 2)
    return imzml_size, ibd_size


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

    def sort_dataset_chunk(sp_inds_list, mzs_list, ints_list):
        mzs = np.concatenate(mzs_list)
        by_mz = np.argsort(mzs)
        sp_mz_int_buf = pd.DataFrame({
            'mz': mzs[by_mz],
            'int': np.concatenate(ints_list)[by_mz].astype(np.float32),
            'sp_i': np.concatenate(sp_inds_list)[by_mz],
        })
        return sp_mz_int_buf

    curr_sp_i = 0
    sp_inds_list, mzs_list, ints_list = [], [], []

    estimated_size_mb = 0
    for x, y in coordinates:
        mzs_, ints_ = imzml_parser.getspectrum(curr_sp_i)
        ints_ = ints_.astype(np.float32)
        sp_idx = np.ones_like(mzs_, dtype=np.uint32) * sp_id_to_idx[curr_sp_i]
        mzs_list.append(mzs_)
        ints_list.append(ints_)
        sp_inds_list.append(sp_idx)
        estimated_size_mb += mzs_.nbytes + ints_.nbytes + sp_idx.nbytes
        curr_sp_i += 1
        if estimated_size_mb > max_size:
            yield sort_dataset_chunk(sp_inds_list, mzs_list, ints_list)
            sp_inds_list, mzs_list, ints_list = [], [], []
            estimated_size_mb = 0

    if len(sp_inds_list) > 0:
        yield sort_dataset_chunk(sp_inds_list, mzs_list, ints_list)


def define_ds_segments(imzml_parser, sp_n, ds_segm_size_mb=5, sample_sp_n=1000):
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

    row_size = (4 if imzml_parser.mzPrecision == 'f' else 8) + 4 + 4
    segm_n = int(np.ceil(total_n_mz * row_size / (ds_segm_size_mb * 2 ** 20)))

    segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n + 1)]
    segm_lower_bounds = [np.quantile(spectra_mzs, q) for q in segm_bounds_q]
    ds_segments_bounds = np.array(list(zip(segm_lower_bounds[:-1], segm_lower_bounds[1:])))

    max_mz_value = 10 ** 5
    ds_segments_bounds[0, 0] = 0
    ds_segments_bounds[-1, 1] = max_mz_value

    return ds_segments_bounds


def segment_spectra_chunk(sp_mz_int_buf, ds_segments_bounds, ds_segments_path):
    def _segment(args):
        segm_i, (l, r) = args
        segm_start, segm_end = np.searchsorted(sp_mz_int_buf.mz.values, (l, r))
        segm = sp_mz_int_buf.iloc[segm_start:segm_end]
        segm.to_msgpack(ds_segments_path / f'ds_segm_{segm_i:04}.msgpack', append=True)
        return segm_i, len(segm)

    with ThreadPoolExecutor() as pool:
        return list(pool.map(_segment, enumerate(ds_segments_bounds)))


def upload_segments(storage, ds_segments_path, segments_n):
    def _upload(segm_i):
        return storage.put_cobject((ds_segments_path / f'ds_segm_{segm_i:04}.msgpack').open('rb'))

    with ThreadPoolExecutor() as pool:
        ds_segms_cobjects = list(pool.map(_upload, range(segments_n)))

    assert len(ds_segms_cobjects) == len(set(co.key for co in ds_segms_cobjects)), 'Duplicate CloudObjects in ds_segms_cobjects'

    return ds_segms_cobjects


def load_and_split_ds_vm(storage, ds_config, ds_segm_size_mb):
    start = time()

    with TemporaryDirectory() as tmp_dir:
        imzml_dir = Path(tmp_dir) / 'imzml'
        imzml_dir.mkdir()
        segments_dir = Path(tmp_dir) / 'segments'
        segments_dir.mkdir()

        logger.debug('Downloading dataset...')
        t = time()
        imzml_size, ibd_size = download_dataset(ds_config['imzml_path'], ds_config['ibd_path'], imzml_dir, storage)
        logger.info('Downloaded dataset in {:.2f} sec'.format(time() - t))
        logger.debug(' * imzML size: {:.2f} mb'.format(imzml_size))
        logger.debug(' * ibd size: {:.2f} mb'.format(ibd_size))

        logger.debug('Loading parser...')
        t = time()
        imzml_parser = ImzMLParser(ds_imzml_path(imzml_dir))
        imzml_reader = imzml_parser.portable_spectrum_reader()
        logger.info('Loaded parser in {:.2f} sec'.format(time() - t))

        coordinates = [coo[:2] for coo in imzml_parser.coordinates]
        sp_n = len(coordinates)

        logger.debug('Defining segments bounds...')
        t = time()
        ds_segments_bounds = define_ds_segments(imzml_parser, sp_n, ds_segm_size_mb=ds_segm_size_mb)
        segments_n = len(ds_segments_bounds)
        logger.info('Defined segments in {:.2f} sec'.format(time() - t))
        logger.debug(' * segments number:', segments_n)

        logger.debug('Segmenting...')
        t = time()
        segm_sizes = []
        for sp_mz_int_buf in parse_dataset_chunks(imzml_parser, coordinates):
            chunk_segm_sizes = segment_spectra_chunk(sp_mz_int_buf, ds_segments_bounds, segments_dir)
            segm_sizes.extend(chunk_segm_sizes)
        ds_segms_len = (
             pd.DataFrame(segm_sizes, columns=['segm_i', 'segm_size'])
            .groupby('segm_i')
            .segm_size
            .sum()
            .sort_index()
            .values
        )

        logger.info('Segmented dataset in {:.2f} sec'.format(time() - t))

        logger.debug('Uploading segments...')
        t = time()
        ds_segms_cobjects = upload_segments(storage, segments_dir, segments_n)
        logger.info('Uploaded segments in {:.2f} sec'.format(time() - t))

        logger.info('load_and_split_ds_vm total: {:.2f} sec'.format(time() - start))

        return imzml_reader, ds_segments_bounds, ds_segms_cobjects, ds_segms_len

