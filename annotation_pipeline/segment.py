import pickle

import numpy as np
import pandas as pd
import sys

import requests
from pyimzml.ImzMLParser import ImzMLParser

from annotation_pipeline.utils import logger, get_pixel_indices, append_pywren_stats, list_keys, \
    clean_from_cos, read_object_with_retry, read_ranges_from_url
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack

ISOTOPIC_PEAK_N = 4
MAX_MZ_VALUE = 10 ** 5


def get_imzml_reader(pw, bucket, input_data):
    def get_portable_imzml_reader(storage):
        imzml_stream = requests.get(input_data['imzml_path'], stream=True).raw
        parser = ImzMLParser(imzml_stream, ibd_file=None)
        imzml_reader = parser.portable_spectrum_reader()
        storage.put_object(Bucket=bucket,
                           Key=input_data["ds_imzml_reader"],
                           Body=pickle.dumps(imzml_reader))
        return imzml_reader

    memory_capacity_mb = 1024
    future = pw.call_async(get_portable_imzml_reader, [])
    imzml_reader = pw.get_result(future)
    append_pywren_stats(future, memory_mb=memory_capacity_mb)

    return imzml_reader


def get_spectra(ibd_url, imzml_reader, sp_inds):
    mz_starts = np.array(imzml_reader.mzOffsets)[sp_inds]
    mz_ends = mz_starts + np.array(imzml_reader.mzLengths)[sp_inds] * np.dtype(imzml_reader.mzPrecision).itemsize
    mz_ranges = np.stack([mz_starts, mz_ends], axis=1)
    int_starts = np.array(imzml_reader.intensityOffsets)[sp_inds]
    int_ends = int_starts + np.array(imzml_reader.intensityLengths)[sp_inds] * np.dtype(imzml_reader.intensityPrecision).itemsize
    int_ranges = np.stack([int_starts, int_ends], axis=1)
    ranges_to_read = np.vstack([mz_ranges, int_ranges])
    data_ranges = read_ranges_from_url(ibd_url, ranges_to_read)
    mz_data = data_ranges[:len(sp_inds)]
    int_data = data_ranges[len(sp_inds):]
    del data_ranges

    for i, sp_idx in enumerate(sp_inds):
        mzs = np.frombuffer(mz_data[i], dtype=imzml_reader.mzPrecision)
        ints = np.frombuffer(int_data[i], dtype=imzml_reader.intensityPrecision)
        mz_data[i] = int_data[i] = None  # Avoid holding memory longer than necessary
        yield sp_idx, mzs, ints


def chunk_spectra(pw, config, input_data, imzml_reader):
    MAX_CHUNK_SIZE = 512 * 1024 ** 2  # 512MB

    sp_id_to_idx = get_pixel_indices(imzml_reader.coordinates)
    row_size = 3 * max(4,
                       np.dtype(imzml_reader.mzPrecision).itemsize,
                       np.dtype(imzml_reader.intensityPrecision).itemsize)

    def plan_chunks():
        chunk_sp_inds = []

        estimated_size_mb = 0
        # Iterate in the same order that intensities are laid out in the file, hopefully this will
        # prevent fragmented read patterns
        for sp_i in np.argsort(imzml_reader.intensityOffsets):
            spectrum_size = imzml_reader.mzLengths[sp_i] * row_size
            if estimated_size_mb + spectrum_size > MAX_CHUNK_SIZE:
                estimated_size_mb = 0
                yield np.array(chunk_sp_inds)
                chunk_sp_inds = []

            estimated_size_mb += spectrum_size
            chunk_sp_inds.append(sp_i)

        if chunk_sp_inds:
            yield np.array(chunk_sp_inds)

    def upload_chunk(storage, ch_i):
        chunk_sp_inds = chunks[ch_i]
        # Get imzml_reader from COS because it's too big to include via pywren captured vars
        imzml_reader = pickle.loads(read_object_with_retry(storage, config["storage"]["ds_bucket"],
                                              input_data["ds_imzml_reader"]))
        n_spectra = sum(imzml_reader.mzLengths[sp_i] for sp_i in chunk_sp_inds)
        sp_mz_int_buf = np.zeros((n_spectra, 3), dtype=imzml_reader.mzPrecision)

        chunk_start = 0
        for sp_i, mzs, ints in get_spectra(input_data['ibd_path'], imzml_reader, chunk_sp_inds):
            chunk_end = chunk_start + len(mzs)
            sp_mz_int_buf[chunk_start:chunk_end, 0] = sp_id_to_idx[sp_i]
            sp_mz_int_buf[chunk_start:chunk_end, 1] = mzs
            sp_mz_int_buf[chunk_start:chunk_end, 2] = ints
            chunk_start = chunk_end

        by_mz = np.argsort(sp_mz_int_buf[:, 1])
        sp_mz_int_buf = sp_mz_int_buf[by_mz]
        del by_mz

        chunk = msgpack.dumps(sp_mz_int_buf)
        key = f'{input_data["ds_chunks"]}/{ch_i}.msgpack'
        size = sys.getsizeof(chunk) * (1 / 1024 ** 2)
        logger.info(f'Uploading spectra chunk {ch_i} - %.2f MB' % size)
        storage.put_object(Bucket=config["storage"]["ds_bucket"],
                           Key=key,
                           Body=chunk)
        logger.info(f'Spectra chunk {ch_i} finished')
        return key

    chunks = list(plan_chunks())
    memory_capacity_mb = 3072
    futures = pw.map(upload_chunk, range(len(chunks)), runtime_memory=memory_capacity_mb)
    pw.wait(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, plus_objects=len(chunks))

    logger.info(f'Uploaded {len(chunks)} dataset chunks')


def define_ds_segments(pw, ibd_url, bucket, ds_imzml_reader, ds_segm_size_mb, sample_n):
    def get_segm_bounds(storage):
        imzml_reader = pickle.loads(read_object_with_retry(storage, bucket, ds_imzml_reader))
        sp_n = len(imzml_reader.coordinates)
        sample_sp_inds = np.random.choice(np.arange(sp_n), min(sp_n, sample_n))
        print(f'Sampling {len(sample_sp_inds)} spectra')
        spectra_sample = list(get_spectra(ibd_url, imzml_reader, sample_sp_inds))

        spectra_mzs = np.concatenate([mzs for sp_id, mzs, ints in spectra_sample])
        print(f'Got {len(spectra_mzs)} mzs')

        total_size = 3 * spectra_mzs.nbytes * sp_n / len(sample_sp_inds)

        segm_n = int(np.ceil(total_size / (ds_segm_size_mb * 2 ** 20)))

        segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n + 1)]
        segm_lower_bounds = [np.quantile(spectra_mzs, q) for q in segm_bounds_q]
        return np.array(list(zip(segm_lower_bounds[:-1], segm_lower_bounds[1:])))

    logger.info('Defining dataset segments bounds')
    memory_capacity_mb = 1024
    future = pw.call_async(get_segm_bounds, [], runtime_memory=memory_capacity_mb)
    ds_segments = pw.get_result(future)
    append_pywren_stats(future, memory_mb=memory_capacity_mb)
    logger.info(f'Generated {len(ds_segments)} dataset bounds: {ds_segments[0]}...{ds_segments[-1]}')
    return ds_segments


def segment_spectra(pw, bucket, ds_chunks_prefix, ds_segments_bounds, ds_segm_size_mb):
    ds_segm_n = len(ds_segments_bounds)

    # extend boundaries of the first and last segments
    # to include all mzs outside of the spectra sample mz range
    ds_segments_bounds = ds_segments_bounds.copy()
    ds_segments_bounds[0, 0] = 0
    ds_segments_bounds[-1, 1] = MAX_MZ_VALUE

    # define first level segmentation and then segment each one into desired number
    first_level_segm_size_mb = 512
    first_level_segm_n = (len(ds_segments_bounds) * ds_segm_size_mb) // first_level_segm_size_mb
    first_level_segm_n = max(first_level_segm_n, 1)
    ds_segments_bounds = np.array_split(ds_segments_bounds, first_level_segm_n)

    def segment_spectra_chunk(obj, storage):
        print(f'Segmenting spectra chunk {obj.key}')
        sp_mz_int_buf = msgpack.loads(obj.data_stream.read())

        def _first_level_segment_upload(segm_i):
            l = ds_segments_bounds[segm_i][0, 0]
            r = ds_segments_bounds[segm_i][-1, 1]
            segm_start, segm_end = np.searchsorted(sp_mz_int_buf[:, 1], (l, r))  # mz expected to be in column 1
            segm = sp_mz_int_buf[segm_start:segm_end]
            return storage.put_cobject(msgpack.dumps(segm))

        with ThreadPoolExecutor(max_workers=128) as pool:
            segms_cobjects = list(pool.map(_first_level_segment_upload, range(len(ds_segments_bounds))))

        return segms_cobjects

    memory_safe_mb = 1024
    memory_capacity_mb = first_level_segm_size_mb * 2 + memory_safe_mb
    first_futures = pw.map(segment_spectra_chunk, f'cos://{bucket}/{ds_chunks_prefix}/', runtime_memory=memory_capacity_mb)
    first_level_segms_cobjects = pw.get_result(first_futures)
    if not isinstance(first_futures, list): first_futures = [first_futures]
    append_pywren_stats(first_futures, memory_mb=memory_capacity_mb, plus_objects=len(first_futures) * len(ds_segments_bounds))

    def merge_spectra_chunk_segments(segm_cobjects, id, storage):
        print(f'Merging segment {id} spectra chunks')

        def _merge(ch_i):
            segm_spectra_chunk = msgpack.loads(storage.get_cobject(segm_cobjects[ch_i]))
            return segm_spectra_chunk

        with ThreadPoolExecutor(max_workers=128) as pool:
            segm = list(pool.map(_merge, range(len(segm_cobjects))))

        segm = np.concatenate(segm)

        # Alternative in-place sorting (slower) :
        # segm.view(f'{segm_dtype},{segm_dtype},{segm_dtype}').sort(order=['f1'], axis=0)
        segm = segm[segm[:, 1].argsort()]

        bounds_list = ds_segments_bounds[id]

        segms_len = []
        segms_cobjects = []
        for segm_j in range(len(bounds_list)):
            l, r = bounds_list[segm_j]
            segm_start, segm_end = np.searchsorted(segm[:, 1], (l, r))  # mz expected to be in column 1
            sub_segm = segm[segm_start:segm_end]
            segms_len.append(len(sub_segm))
            base_id = sum([len(bounds) for bounds in ds_segments_bounds[:id]])
            segm_i = base_id + segm_j
            print(f'Storing dataset segment {segm_i}')
            segms_cobjects.append(storage.put_cobject(msgpack.dumps(sub_segm)))

        return segms_len, segms_cobjects

    second_level_segms_cobjects = np.transpose(first_level_segms_cobjects).tolist()
    second_level_segms_cobjects = [[segm_cobjects] for segm_cobjects in second_level_segms_cobjects]

    # same memory capacity
    second_futures = pw.map(merge_spectra_chunk_segments, second_level_segms_cobjects, runtime_memory=memory_capacity_mb)
    ds_segms_len, ds_segms_cobjects = list(zip(*pw.get_result(second_futures)))
    ds_segms_len = list(np.concatenate(ds_segms_len))
    ds_segms_cobjects = list(np.concatenate(ds_segms_cobjects))
    append_pywren_stats(second_futures, memory_mb=memory_capacity_mb, plus_objects=ds_segm_n, minus_objects=len(first_futures) * len(ds_segments_bounds))

    return ds_segms_cobjects, ds_segms_len


def clip_centr_df(pw, bucket, centr_chunks_prefix, clip_centr_chunk_prefix, mz_min, mz_max):
    def clip_centr_df_chunk(obj, id, storage):
        print(f'Clipping centroids dataframe chunk {obj.key}')
        centroids_df_chunk = pd.read_msgpack(obj.data_stream._raw_stream).sort_values('mz')
        centroids_df_chunk = centroids_df_chunk[centroids_df_chunk.mz > 0]

        ds_mz_range_unique_formulas = centroids_df_chunk[(mz_min < centroids_df_chunk.mz) &
                                                         (centroids_df_chunk.mz < mz_max)].index.unique()
        centr_df_chunk = centroids_df_chunk[centroids_df_chunk.index.isin(ds_mz_range_unique_formulas)].reset_index()
        storage.put_object(Bucket=bucket,
                           Key=f'{clip_centr_chunk_prefix}/{id}.msgpack',
                           Body=centr_df_chunk.to_msgpack())

        return centr_df_chunk.shape[0]

    memory_capacity_mb = 512
    futures = pw.map(clip_centr_df_chunk, f'cos://{bucket}/{centr_chunks_prefix}/', runtime_memory=memory_capacity_mb)
    centr_n = sum(pw.get_result(futures))
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, plus_objects=len(futures))

    logger.info(f'Prepared {centr_n} centroids')
    return centr_n


def define_centr_segments(pw, bucket, clip_centr_chunk_prefix, centr_n, ds_segm_n, ds_segm_size_mb):
    logger.info('Defining centroids segments bounds')

    def get_first_peak_mz(obj):
        print(f'Extracting first peak mz values from clipped centroids dataframe {obj.key}')
        centr_df = pd.read_msgpack(obj.data_stream._raw_stream)
        first_peak_df = centr_df[centr_df.peak_i == 0]
        return first_peak_df.mz.values

    memory_capacity_mb = 512
    futures = pw.map(get_first_peak_mz, f'cos://{bucket}/{clip_centr_chunk_prefix}/', runtime_memory=memory_capacity_mb)
    first_peak_df_mz = np.concatenate(pw.get_result(futures))
    append_pywren_stats(futures, memory_mb=memory_capacity_mb)

    ds_size_mb = ds_segm_n * ds_segm_size_mb
    data_per_centr_segm_mb = 50
    peaks_per_centr_segm = 1e4
    centr_segm_n = int(max(ds_size_mb // data_per_centr_segm_mb, centr_n // peaks_per_centr_segm, 32))

    segm_bounds_q = [i * 1 / centr_segm_n for i in range(0, centr_segm_n)]
    centr_segm_lower_bounds = np.quantile(first_peak_df_mz, segm_bounds_q)

    logger.info(f'Generated {len(centr_segm_lower_bounds)} centroids bounds: {centr_segm_lower_bounds[0]}...{centr_segm_lower_bounds[-1]}')
    return centr_segm_lower_bounds


def segment_centroids(pw, bucket, clip_centr_chunk_prefix, centr_segm_prefix, centr_segm_lower_bounds):
    centr_segm_n = len(centr_segm_lower_bounds)
    centr_segm_lower_bounds = centr_segm_lower_bounds.copy()

    # define first level segmentation and then segment each one into desired number
    first_level_centr_segm_n = min(32, len(centr_segm_lower_bounds))
    centr_segm_lower_bounds = np.array_split(centr_segm_lower_bounds, first_level_centr_segm_n)
    first_level_centr_segm_bounds = np.array([bounds[0] for bounds in centr_segm_lower_bounds])

    def segment_centr_df(centr_df, db_segm_lower_bounds):
        first_peak_df = centr_df[centr_df.peak_i == 0].copy()
        segment_mapping = np.searchsorted(db_segm_lower_bounds, first_peak_df.mz.values, side='right') - 1
        first_peak_df['segm_i'] = segment_mapping
        centr_segm_df = pd.merge(centr_df, first_peak_df[['formula_i', 'segm_i']], on='formula_i').sort_values('mz')
        return centr_segm_df

    def segment_centr_chunk(obj, id, storage):
        print(f'Segmenting clipped centroids dataframe chunk {obj.key}')
        centr_df = pd.read_msgpack(obj.data_stream._raw_stream)
        centr_segm_df = segment_centr_df(centr_df, first_level_centr_segm_bounds)

        def _first_level_upload(args):
            segm_i, df = args
            storage.put_object(Bucket=bucket,
                               Key=f'{centr_segm_prefix}/chunk/{segm_i}/{id}.msgpack',
                               Body=df.to_msgpack())

        with ThreadPoolExecutor(max_workers=128) as pool:
            pool.map(_first_level_upload, [(segm_i, df) for segm_i, df in centr_segm_df.groupby('segm_i')])

    memory_capacity_mb = 512
    first_futures = pw.map(segment_centr_chunk, f'cos://{bucket}/{clip_centr_chunk_prefix}/', runtime_memory=memory_capacity_mb)
    pw.get_result(first_futures)
    append_pywren_stats(first_futures, memory_mb=memory_capacity_mb,
                        plus_objects=len(first_futures) * len(centr_segm_lower_bounds))

    def merge_centr_df_segments(segm_i, storage):
        print(f'Merging segment {segm_i} clipped centroids chunks')

        keys = list_keys(bucket, f'{centr_segm_prefix}/chunk/{segm_i}/', storage)

        def _merge(key):
            segm_centr_df_chunk = read_object_with_retry(storage, bucket, key, pd.read_msgpack)
            return segm_centr_df_chunk

        with ThreadPoolExecutor(max_workers=128) as pool:
            segm = pd.concat(list(pool.map(_merge, keys)))
            del segm['segm_i']

        clean_from_cos(None, bucket, f'{centr_segm_prefix}/chunk/{segm_i}/', storage)
        centr_segm_df = segment_centr_df(segm, centr_segm_lower_bounds[segm_i])

        def _second_level_upload(args):
            segm_j, df = args
            base_id = sum([len(bounds) for bounds in centr_segm_lower_bounds[:segm_i]])
            id = base_id + segm_j
            print(f'Storing centroids segment {id}')
            storage.put_object(Bucket=bucket,
                               Key=f'{centr_segm_prefix}/{id}.msgpack',
                               Body=df.to_msgpack())

        with ThreadPoolExecutor(max_workers=128) as pool:
            pool.map(_second_level_upload, [(segm_i, df) for segm_i, df in centr_segm_df.groupby('segm_i')])

    memory_capacity_mb = 1024
    second_futures = pw.map(merge_centr_df_segments, range(len(centr_segm_lower_bounds)), runtime_memory=memory_capacity_mb)
    pw.get_result(second_futures)
    append_pywren_stats(second_futures, memory_mb=memory_capacity_mb,
                        plus_objects=centr_segm_n, minus_objects=len(first_futures) * len(centr_segm_lower_bounds))

    return centr_segm_n
