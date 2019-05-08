from shutil import rmtree
import numpy as np
import pandas as pd

from .utils import logger, get_pixel_indices

ISOTOPIC_PEAK_N = 4
MAX_MZ_VALUE = 10**5


def spectra_sample_gen(imzml_parser, sample_ratio=0.05):
    sp_n = len(imzml_parser.coordinates)
    sample_size = int(sp_n * sample_ratio)
    sample_sp_inds = np.random.choice(np.arange(sp_n), sample_size)
    for sp_idx in sample_sp_inds:
        mzs, ints = imzml_parser.getspectrum(sp_idx)
        yield sp_idx, mzs, ints


def define_ds_segments(imzml_parser, ds_segm_size_mb=5, sample_ratio=0.05):
    logger.info(f'Defining dataset segment bounds')
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


def segment_spectra_chunk(sp_mz_int_buf, mz_segments, ds_segments_path):
    for segm_i, (l, r) in mz_segments:
        segm_start, segm_end = np.searchsorted(sp_mz_int_buf[:, 1], (l, r))  # mz expected to be in column 1
        pd.to_msgpack(ds_segments_path / f'ds_segm_{segm_i:04}.msgpack',
                      sp_mz_int_buf[segm_start:segm_end],
                      append=True)


def segment_spectra(imzml_parser, coordinates, ds_segments, ds_segments_path):

    def chunk_list(l, size=5000):
        n = (len(l) - 1) // size + 1
        for i in range(n):
            yield l[size * i:size * (i + 1)]

    logger.info(f'Segmenting dataset into {len(ds_segments)} segments')

    rmtree(ds_segments_path, ignore_errors=True)
    ds_segments_path.mkdir(parents=True)

    # extend boundaries of the first and last segments
    # to include all mzs outside of the spectra sample mz range
    mz_segments = ds_segments.copy()
    mz_segments[0, 0] = 0
    mz_segments[-1, 1] = MAX_MZ_VALUE
    mz_segments = list(enumerate(mz_segments))

    sp_id_to_idx = get_pixel_indices(coordinates)

    chunk_size = 5000
    coord_chunk_it = chunk_list(coordinates, chunk_size)

    sp_i = 0
    sp_inds_list, mzs_list, ints_list = [], [], []
    for ch_i, coord_chunk in enumerate(coord_chunk_it):
        logger.debug(f'Segmenting spectra chunk {ch_i}')

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
        segment_spectra_chunk(sp_mz_int_buf, mz_segments, ds_segments_path)

        sp_inds_list, mzs_list, ints_list = [], [], []


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
    logger.info(f'Segmenting centroids into {segm_n} segments')

    rmtree(centr_segm_path, ignore_errors=True)
    centr_segm_path.mkdir(parents=True)

    first_peak_df = centr_df[centr_df.peak_i == 0].copy()
    segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n)]
    segm_lower_bounds = list(np.quantile(first_peak_df.mz, q) for q in segm_bounds_q)

    segment_mapping = np.searchsorted(segm_lower_bounds, first_peak_df.mz.values, side='right') - 1
    first_peak_df['segm_i'] = segment_mapping

    centr_segm_df = pd.merge(centr_df, first_peak_df[['formula_i', 'segm_i']],
                             on='formula_i').sort_values('mz')
    for segm_i, df in centr_segm_df.groupby('segm_i'):
        pd.to_msgpack(f'{centr_segm_path}/centr_segm_{segm_i:04}.msgpack', df)


