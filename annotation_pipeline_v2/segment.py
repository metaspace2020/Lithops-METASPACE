import numpy as np
import pandas as pd
import pickle
import sys
from .utils import logger, get_pixel_indices, get_ibm_cos_client
from concurrent.futures import ThreadPoolExecutor
import pywren_ibm_cloud as pywren
import msgpack_numpy as msgpack

ISOTOPIC_PEAK_N = 4
MAX_MZ_VALUE = 10**5


def chunk_spectra(config, input_data, sp_n, imzml_parser, coordinates):

    def chunk_list(l, size=5000):
        n = (len(l) - 1) // size + 1
        for i in range(n):
            yield l[size * i:size * (i + 1)]

    cos_client = get_ibm_cos_client(config)

    sp_id_to_idx = get_pixel_indices(coordinates)

    chunk_size = 5000
    coord_chunk_it = chunk_list(coordinates, chunk_size)

    sp_i_lower_bounds = []
    sp_i_bound = 0
    while sp_i_bound <= sp_n:
        sp_i_lower_bounds.append(sp_i_bound)
        sp_i_bound += chunk_size

    logger.info(f'Parsing dataset into {len(sp_i_lower_bounds)} chunks')
    keys = [f'{input_data["ds_chunks"]}/{ch_i}.msgpack' for ch_i in range(len(sp_i_lower_bounds))]

    for ch_i, coord_chunk in enumerate(coord_chunk_it):
        logger.info(f'Parsing spectra chunk {ch_i}')
        sp_i = sp_i_lower_bounds[ch_i]
        sp_inds_list, mzs_list, ints_list = [], [], []
        for x, y in coord_chunk:
            mzs_, ints_ = imzml_parser.getspectrum(sp_i)
            mzs_, ints_ = map(np.array, [mzs_, ints_])
            sp_idx = sp_id_to_idx[sp_i]
            sp_inds_list.append(np.ones_like(mzs_) * sp_idx)
            mzs_list.append(mzs_)
            ints_list.append(ints_)
            sp_i += 1

        dtype = imzml_parser.mzPrecision
        mzs = np.concatenate(mzs_list)
        by_mz = np.argsort(mzs)
        sp_mz_int_buf = np.array([np.concatenate(sp_inds_list)[by_mz],
                                  mzs[by_mz],
                                  np.concatenate(ints_list)[by_mz]], dtype).T

        chunk = msgpack.dumps(sp_mz_int_buf)
        size = sys.getsizeof(chunk)*(1/1024**2)
        logger.info(f'Uploading spectra chunk {ch_i} - %.2f MB' % size)
        cos_client.put_object(Bucket=config["storage"]["ds_bucket"],
                              Key=keys[ch_i],
                              Body=chunk)
        logger.info(f'Spectra chunk {ch_i} finished')

    return keys


def spectra_sample_gen(imzml_parser, sample_ratio=0.05):
    sp_n = len(imzml_parser.coordinates)
    sample_size = int(sp_n * sample_ratio)
    sample_sp_inds = np.random.choice(np.arange(sp_n), sample_size)
    for sp_idx in sample_sp_inds:
        mzs, ints = imzml_parser.getspectrum(sp_idx)
        yield sp_idx, mzs, ints


def define_ds_segments(imzml_parser, ds_segm_size_mb=5, sample_ratio=0.05):
    logger.info('Defining dataset segment bounds')
    spectra_sample = list(spectra_sample_gen(imzml_parser, sample_ratio=sample_ratio))

    spectra_mzs = np.array([mz for sp_id, mzs, ints in spectra_sample for mz in mzs])
    total_n_mz = spectra_mzs.shape[0] / sample_ratio

    float_prec = 4 if imzml_parser.mzPrecision == 'f' else 8
    segm_arr_columns = 3
    segm_n = segm_arr_columns * (total_n_mz * float_prec) // (ds_segm_size_mb * 2**20)
    segm_n = max(1, int(segm_n))

    segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n + 1)]
    segm_lower_bounds = [np.quantile(spectra_mzs, q) for q in segm_bounds_q]
    ds_segments = np.array(list(zip(segm_lower_bounds[:-1], segm_lower_bounds[1:])))

    logger.info(f'Generated {len(ds_segments)} dataset segments: {ds_segments[0]}...{ds_segments[-1]}')
    return ds_segments


def segment_spectra(config, bucket, ds_chunks_prefix, ds_segments_prefix, ds_segments_bounds):
    # extend boundaries of the first and last segments
    # to include all mzs outside of the spectra sample mz range
    mz_segments = ds_segments_bounds.copy()
    mz_segments[0, 0] = 0
    mz_segments[-1, 1] = MAX_MZ_VALUE
    mz_segments = list(enumerate(mz_segments))

    def segment_spectra_chunk(bucket, key, data_stream, ibm_cos):
        ch_i = int(key.split("/")[-1].split(".msgpack")[0])
        print(f'Segmenting spectra chunk {ch_i}')
        sp_mz_int_buf = msgpack.loads(data_stream.read())

        def _segment_spectra_chunk(args):
            segm_i, (l, r) = args
            segm_start, segm_end = np.searchsorted(sp_mz_int_buf[:, 1], (l, r))  # mz expected to be in column 1
            segm = sp_mz_int_buf[segm_start:segm_end]
            ibm_cos.put_object(Bucket=bucket,
                               Key=f'{ds_segments_prefix}/chunk/{segm_i}/{ch_i}.msgpack',
                               Body=msgpack.dumps(segm))

        with ThreadPoolExecutor(max_workers=128) as pool:
            pool.map(_segment_spectra_chunk, mz_segments)

    def merge_spectra_chunk_segments(results):

        def _merge(segm_i, ibm_cos):
            print(f'Merging segment {segm_i} spectra chunks')

            objs = ibm_cos.list_objects_v2(Bucket=bucket, Prefix=f'{ds_segments_prefix}/chunk/{segm_i}/')
            if 'Contents' in objs:
                keys = [obj['Key'] for obj in objs['Contents']]

                segm = []
                for key in keys:
                    segm_spectra_chunk = msgpack.loads(ibm_cos.get_object(Bucket=bucket, Key=key)['Body'].read())
                    segm.append(segm_spectra_chunk)

                ibm_cos.put_object(Bucket=bucket,
                                   Key=f'{ds_segments_prefix}/{segm_i}.msgpack',
                                   Body=msgpack.dumps(segm))

                temp_formatted_keys = {'Objects': [{'Key': key} for key in keys]}
                ibm_cos.delete_objects(Bucket=bucket, Delete=temp_formatted_keys)

        pw = pywren.ibm_cf_executor(config=config, runtime_memory=512)
        pw.map(_merge, range(len(mz_segments)))
        pw.get_result()

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=1024)
    pw.map_reduce(segment_spectra_chunk, f'{bucket}/{ds_chunks_prefix}', merge_spectra_chunk_segments)
    pw.get_result()


def clip_centroids_df(key, data_stream, mz_min, mz_max):
    print("Downloading centroids database")
    centroids_df = pickle.loads(data_stream.read()).sort_values('mz')
    centroids_df = centroids_df[centroids_df.mz > 0]

    ds_mz_range_unique_formulas = centroids_df[(mz_min < centroids_df.mz) &
                                               (centroids_df.mz < mz_max)].index.unique()
    centr_df = centroids_df[centroids_df.index.isin(ds_mz_range_unique_formulas)].reset_index()
    print("Saving centr_df")
    pd.to_msgpack('/tmp/centr_df.msgpack', centr_df)
    return centr_df.shape[0]


def calculate_centroids_segments_n(centr_n, ds_segm_n, ds_segm_size_mb):
    ds_size_mb = ds_segm_n * ds_segm_size_mb
    data_per_centr_segm_mb = 50
    peaks_per_centr_segm = 1e4
    centr_segm_n = int(max(ds_size_mb // data_per_centr_segm_mb,
                           centr_n // peaks_per_centr_segm,
                           32))
    return centr_segm_n


def segment_centroids(ds_bucket, db_segm_n, centr_segm_prefix, ibm_cos):
    print("Reading centr_df")
    centr_df = pd.read_msgpack('/tmp/centr_df.msgpack')
    first_peak_df = centr_df[centr_df.peak_i == 0].copy()
    segm_bounds_q = [i * 1 / db_segm_n for i in range(0, db_segm_n)]
    segm_lower_bounds = list(np.quantile(first_peak_df.mz, q) for q in segm_bounds_q)

    segment_mapping = np.searchsorted(segm_lower_bounds, first_peak_df.mz.values, side='right') - 1
    first_peak_df['segm_i'] = segment_mapping

    centr_segm_df = pd.merge(centr_df, first_peak_df[['formula_i', 'segm_i']],
                             on='formula_i').sort_values('mz')

    def upload_db_segment(args):
        segm_i, df = args
        ibm_cos.put_object(Bucket=ds_bucket,
                           Key=f'{centr_segm_prefix}/{segm_i}.msgpack',
                           Body=df.to_msgpack())

    print("Segmenting centroids")
    with ThreadPoolExecutor(max_workers=128) as pool:
        pool.map(upload_db_segment, [(segm_i, df) for segm_i, df in centr_segm_df.groupby('segm_i')])
