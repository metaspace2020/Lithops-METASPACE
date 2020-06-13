import pickle

import numpy as np
import pandas as pd
import sys
import math

import requests
from pyimzml.ImzMLParser import ImzMLParser

from annotation_pipeline.utils import logger, get_pixel_indices, append_pywren_stats, read_object_with_retry, \
    read_cloud_object_with_retry, read_ranges_from_url
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack

ISOTOPIC_PEAK_N = 4
MAX_MZ_VALUE = 10 ** 5


def get_imzml_reader(pw, imzml_path):
    def get_portable_imzml_reader(storage):
        imzml_stream = requests.get(imzml_path, stream=True).raw
        parser = ImzMLParser(imzml_stream, ibd_file=None)
        imzml_reader = parser.portable_spectrum_reader()
        imzml_cobject = storage.put_cobject(pickle.dumps(imzml_reader))
        return imzml_reader, imzml_cobject

    memory_capacity_mb = 1024
    future = pw.call_async(get_portable_imzml_reader, [])
    imzml_reader, imzml_cobject = pw.get_result(future)
    append_pywren_stats(future, memory_mb=memory_capacity_mb, cloud_objects_n=1)

    return imzml_reader, imzml_cobject


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


def chunk_spectra(pw, ibd_path, imzml_cobject, imzml_reader):
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

    def upload_chunk(ch_i, storage):
        chunk_sp_inds = chunks[ch_i]
        # Get imzml_reader from COS because it's too big to include via pywren captured vars
        imzml_reader = pickle.loads(read_cloud_object_with_retry(storage, imzml_cobject))
        n_spectra = sum(imzml_reader.mzLengths[sp_i] for sp_i in chunk_sp_inds)
        sp_mz_int_buf = np.zeros((n_spectra, 3), dtype=imzml_reader.mzPrecision)

        chunk_start = 0
        for sp_i, mzs, ints in get_spectra(ibd_path, imzml_reader, chunk_sp_inds):
            chunk_end = chunk_start + len(mzs)
            sp_mz_int_buf[chunk_start:chunk_end, 0] = sp_id_to_idx[sp_i]
            sp_mz_int_buf[chunk_start:chunk_end, 1] = mzs
            sp_mz_int_buf[chunk_start:chunk_end, 2] = ints
            chunk_start = chunk_end

        by_mz = np.argsort(sp_mz_int_buf[:, 1])
        sp_mz_int_buf = sp_mz_int_buf[by_mz]
        del by_mz

        chunk = msgpack.dumps(sp_mz_int_buf)
        size = sys.getsizeof(chunk) * (1 / 1024 ** 2)
        logger.info(f'Uploading spectra chunk {ch_i} - %.2f MB' % size)
        chunk_cobject = storage.put_cobject(chunk)
        logger.info(f'Spectra chunk {ch_i} finished')
        return chunk_cobject

    chunks = list(plan_chunks())
    memory_capacity_mb = 3072
    futures = pw.map(upload_chunk, range(len(chunks)), runtime_memory=memory_capacity_mb)
    ds_chunks_cobjects = pw.get_result(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(chunks))

    return ds_chunks_cobjects


def define_ds_segments(pw, ibd_url, imzml_cobject, ds_segm_size_mb, sample_n):
    def get_segm_bounds(storage):
        imzml_reader = pickle.loads(read_cloud_object_with_retry(storage, imzml_cobject))
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
    return ds_segments


def segment_spectra(pw, ds_chunks_cobjects, ds_segments_bounds, ds_segm_size_mb, ds_segm_dtype):
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

    def segment_spectra_chunk(chunk_cobject, id, storage):
        print(f'Segmenting spectra chunk {id}')
        sp_mz_int_buf = read_cloud_object_with_retry(storage, chunk_cobject, msgpack.load)

        def _first_level_segment_upload(segm_i):
            l = ds_segments_bounds[segm_i][0, 0]
            r = ds_segments_bounds[segm_i][-1, 1]
            segm_start, segm_end = np.searchsorted(sp_mz_int_buf[:, 1], (l, r))  # mz expected to be in column 1
            segm = sp_mz_int_buf[segm_start:segm_end]
            return storage.put_cobject(msgpack.dumps(segm))

        with ThreadPoolExecutor(max_workers=128) as pool:
            sub_segms_cobjects = list(pool.map(_first_level_segment_upload, range(len(ds_segments_bounds))))

        return sub_segms_cobjects

    memory_safe_mb = 1536
    memory_capacity_mb = first_level_segm_size_mb * 2 + memory_safe_mb
    first_futures = pw.map(segment_spectra_chunk, ds_chunks_cobjects, runtime_memory=memory_capacity_mb)
    first_level_segms_cobjects = pw.get_result(first_futures)
    if not isinstance(first_futures, list): first_futures = [first_futures]
    append_pywren_stats(first_futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(first_futures) * len(ds_segments_bounds))

    def merge_spectra_chunk_segments(segm_cobjects, id, storage):
        print(f'Merging segment {id} spectra chunks')

        def _merge(ch_i):
            segm_spectra_chunk = read_cloud_object_with_retry(storage, segm_cobjects[ch_i], msgpack.load)
            return segm_spectra_chunk

        with ThreadPoolExecutor(max_workers=128) as pool:
            segm = list(pool.map(_merge, range(len(segm_cobjects))))

        segm = np.concatenate(segm)

        # Alternative in-place sorting (slower) :
        # segm.view(f'{ds_segm_dtype},{ds_segm_dtype},{ds_segm_dtype}').sort(order=['f1'], axis=0)
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
    append_pywren_stats(second_futures, memory_mb=memory_capacity_mb, cloud_objects_n=ds_segm_n)

    return ds_segms_cobjects, ds_segms_len


def clip_centr_df(pw, bucket, centr_chunks_prefix, mz_min, mz_max):
    def clip_centr_df_chunk(obj, storage):
        print(f'Clipping centroids dataframe chunk {obj.key}')
        centroids_df_chunk = pd.read_msgpack(obj.data_stream._raw_stream).sort_values('mz')
        centroids_df_chunk = centroids_df_chunk[centroids_df_chunk.mz > 0]

        ds_mz_range_unique_formulas = centroids_df_chunk[(mz_min < centroids_df_chunk.mz) &
                                                         (centroids_df_chunk.mz < mz_max)].index.unique()
        centr_df_chunk = centroids_df_chunk[centroids_df_chunk.index.isin(ds_mz_range_unique_formulas)].reset_index()
        clip_centr_chunk_cobject = storage.put_cobject(centr_df_chunk.to_msgpack())

        return clip_centr_chunk_cobject, centr_df_chunk.shape[0]

    memory_capacity_mb = 512
    futures = pw.map(clip_centr_df_chunk, f'cos://{bucket}/{centr_chunks_prefix}/', runtime_memory=memory_capacity_mb)
    clip_centr_chunks_cobjects, centr_n = list(zip(*pw.get_result(futures)))
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(futures))

    clip_centr_chunks_cobjects = list(clip_centr_chunks_cobjects)
    centr_n = sum(centr_n)
    logger.info(f'Prepared {centr_n} centroids')
    return clip_centr_chunks_cobjects, centr_n


def define_centr_segments(pw, clip_centr_chunks_cobjects, init_centr_segm_n):
    logger.info('Defining centroids segments bounds')

    def get_first_peak_mz(cobject, id, storage):
        print(f'Extracting first peak mz values from clipped centroids dataframe {id}')
        centr_df = read_cloud_object_with_retry(storage, cobject, pd.read_msgpack)
        first_peak_df = centr_df[centr_df.peak_i == 0]
        return first_peak_df.mz.values

    memory_capacity_mb = 512
    futures = pw.map(get_first_peak_mz, clip_centr_chunks_cobjects, runtime_memory=memory_capacity_mb)
    first_peak_df_mz = np.concatenate(pw.get_result(futures))
    append_pywren_stats(futures, memory_mb=memory_capacity_mb)

    segm_bounds_q = [i * 1 / init_centr_segm_n for i in range(0, init_centr_segm_n)]
    centr_segm_lower_bounds = np.quantile(first_peak_df_mz, segm_bounds_q)

    logger.info(f'Generated {len(centr_segm_lower_bounds)} centroids bounds: {centr_segm_lower_bounds[0]}...{centr_segm_lower_bounds[-1]}')
    return centr_segm_lower_bounds


def segment_centroids(pw, clip_centr_chunks_cobjects, centr_segm_lower_bounds, ds_segms_bounds, ds_segm_size_mb,
                      max_ds_segms_size_per_db_segm_mb, ppm):
    centr_segm_n = len(centr_segm_lower_bounds)
    centr_segm_lower_bounds = centr_segm_lower_bounds.copy()

    def segment_centr_df(centr_df, db_segm_lower_bounds):
        first_peak_df = centr_df[centr_df.peak_i == 0].copy()
        segment_mapping = np.searchsorted(db_segm_lower_bounds, first_peak_df.mz.values, side='right') - 1
        first_peak_df['segm_i'] = segment_mapping
        centr_segm_df = pd.merge(centr_df, first_peak_df[['formula_i', 'segm_i']], on='formula_i').sort_values('mz')
        return centr_segm_df

    def segment_centr_chunk(cobject, id, storage):
        print(f'Segmenting clipped centroids dataframe chunk {id}')
        centr_df = read_cloud_object_with_retry(storage, cobject, pd.read_msgpack)
        centr_segm_df = segment_centr_df(centr_df, centr_segm_lower_bounds)

        def _first_level_upload(args):
            segm_i, df = args
            del df['segm_i']
            return segm_i, storage.put_cobject(df.to_msgpack())

        with ThreadPoolExecutor(max_workers=128) as pool:
            sub_segms = [(segm_i, df) for segm_i, df in centr_segm_df.groupby('segm_i')]
            sub_segms_cobjects = list(pool.map(_first_level_upload, sub_segms))

        return dict(sub_segms_cobjects)

    memory_capacity_mb = 512
    first_futures = pw.map(segment_centr_chunk, clip_centr_chunks_cobjects, runtime_memory=memory_capacity_mb)
    first_level_segms_cobjects = pw.get_result(first_futures)
    append_pywren_stats(first_futures, memory_mb=memory_capacity_mb,
                        cloud_objects_n=len(first_futures) * len(centr_segm_lower_bounds))

    def merge_centr_df_segments(segm_cobjects, id, storage):
        print(f'Merging segment {id} clipped centroids chunks')

        def _merge(cobject):
            segm_centr_df_chunk = read_cloud_object_with_retry(storage, cobject, pd.read_msgpack)
            return segm_centr_df_chunk

        with ThreadPoolExecutor(max_workers=128) as pool:
            segm = pd.concat(list(pool.map(_merge, segm_cobjects)))

        def _second_level_segment(segm, sub_segms_n):
            segm_bounds_q = [i * 1 / sub_segms_n for i in range(0, sub_segms_n)]
            sub_segms_lower_bounds = np.quantile(segm[segm.peak_i == 0].mz.values, segm_bounds_q)
            centr_segm_df = segment_centr_df(segm, sub_segms_lower_bounds)

            sub_segms = []
            for segm_i, df in centr_segm_df.groupby('segm_i'):
                del df['segm_i']
                sub_segms.append(df)
            return sub_segms

        from annotation_pipeline.image import choose_ds_segments
        first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segms_bounds, segm, ppm)
        init_ds_segms_to_download_n = last_ds_segm_i - first_ds_segm_i + 1
        segms = [(init_ds_segms_to_download_n, segm)]
        max_ds_segms_to_download_n, max_segm = segms[0]

        max_iterations_n = 100
        iterations_n = 1
        while max_ds_segms_to_download_n * ds_segm_size_mb > max_ds_segms_size_per_db_segm_mb and iterations_n < max_iterations_n:

            sub_segms = []
            sub_segms_n = math.ceil(max_ds_segms_to_download_n * ds_segm_size_mb / max_ds_segms_size_per_db_segm_mb)
            for sub_segm in _second_level_segment(max_segm, sub_segms_n):
                first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segms_bounds, sub_segm, ppm)
                ds_segms_to_download_n = last_ds_segm_i - first_ds_segm_i + 1
                sub_segms.append((ds_segms_to_download_n, sub_segm))

            segms = sub_segms + segms[1:]
            segms = sorted(segms, key=lambda x: x[0], reverse=True)
            iterations_n += 1
            max_ds_segms_to_download_n, max_segm = segms[0]

        # Alternative approach to test if the while loop didnt converge
        # if iterations_n >= max_iterations_n:
        #     segm_centr_n = segm.shape[0]
        #     ds_size_mb = init_ds_segms_to_download_n * ds_segm_size_mb
        #     data_per_centr_segm_mb = 50
        #     peaks_per_centr_segm = 1e4
        #     sub_segms_n = int(max(ds_size_mb // data_per_centr_segm_mb, segm_centr_n // peaks_per_centr_segm, 32))
        #     segms = list(enumerate(_second_level_segment(segm, sub_segms_n)))

        def _second_level_upload(df):
            return storage.put_cobject(df.to_msgpack())

        print(f'Storing {len(segms)} centroids segments')
        with ThreadPoolExecutor(max_workers=128) as pool:
            segms = [df for _, df in segms]
            segms_cobjects = list(pool.map(_second_level_upload, segms))

        return segms_cobjects

    from collections import defaultdict
    second_level_segms_cobjects = defaultdict(list)
    for sub_segms_cobjects in first_level_segms_cobjects:
        for first_level_segm_i in sub_segms_cobjects:
            second_level_segms_cobjects[first_level_segm_i].append(sub_segms_cobjects[first_level_segm_i])
    second_level_segms_cobjects = sorted(second_level_segms_cobjects.items(), key=lambda x: x[0])
    second_level_segms_cobjects = [[cobjects] for segm_i, cobjects in second_level_segms_cobjects]

    memory_capacity_mb = 2048
    second_futures = pw.map(merge_centr_df_segments, second_level_segms_cobjects, runtime_memory=memory_capacity_mb)
    db_segms_cobjects = list(np.concatenate(pw.get_result(second_futures)))
    append_pywren_stats(second_futures, memory_mb=memory_capacity_mb, cloud_objects_n=centr_segm_n)

    return db_segms_cobjects
