import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack

from annotation_pipeline.utils import ds_dims, get_pixel_indices
from annotation_pipeline.validate import make_compute_image_metrics, formula_image_metrics


def gen_iso_images(sp_inds, sp_mzs, sp_ints, centr_df, nrows, ncols, ppm=3, min_px=1):
    if len(sp_inds) > 0:
        by_sp_mz = np.argsort(sp_mzs)  # sort order by mz ascending
        sp_mzs = sp_mzs[by_sp_mz]
        sp_inds = sp_inds[by_sp_mz]
        sp_ints = sp_ints[by_sp_mz]

        by_centr_mz = np.argsort(centr_df.mz.values)  # sort order by mz ascending
        centr_mzs = centr_df.mz.values[by_centr_mz]
        centr_f_inds = centr_df.formula_i.values[by_centr_mz]
        centr_p_inds = centr_df.peak_i.values[by_centr_mz]
        centr_ints = centr_df.int.values[by_centr_mz]

        lower = centr_mzs - centr_mzs * ppm * 1e-6
        upper = centr_mzs + centr_mzs * ppm * 1e-6
        lower_idx = np.searchsorted(sp_mzs, lower, 'l')
        upper_idx = np.searchsorted(sp_mzs, upper, 'r')

        for i, (l, u) in enumerate(zip(lower_idx, upper_idx)):
            m = None
            if u - l >= min_px:
                data = sp_ints[l:u]
                inds = sp_inds[l:u]
                row_inds = inds / ncols
                col_inds = inds % ncols
                m = coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols), copy=True)
            yield centr_f_inds[i], centr_p_inds[i], centr_ints[i], m


def read_ds_segments(ds_bucket, ds_segm_keys, first_segm_i, last_segm_i, ibm_cos):

    def read_ds_segment(ds_segm_key):
        data_stream = ibm_cos.get_object(Bucket=ds_bucket, Key=ds_segm_key)['Body']
        data = msgpack.loads(data_stream.read())
        if type(data) == list:
            sp_arr = np.concatenate(data)
        else:
            sp_arr = data
        return sp_arr

    with ThreadPoolExecutor(max_workers=128) as pool:
        ds_segm_keys = [ds_segm_keys[segm_i] for segm_i in range(first_segm_i, last_segm_i + 1)]
        sp_arr = list(pool.map(read_ds_segment, ds_segm_keys))

    sp_arr = [a for a in sp_arr if a.shape[0] > 0]
    if len(sp_arr) > 0:
        sp_arr = np.concatenate(sp_arr)
        sp_arr = sp_arr[sp_arr[:, 1].argsort()]  # assume mz in column 1
    else:
        sp_arr = np.empty((0, 3))
    return sp_arr


def make_sample_area_mask(coordinates):
    pixel_indices = get_pixel_indices(coordinates)
    nrows, ncols = ds_dims(coordinates)
    sample_area_mask = np.zeros(ncols * nrows, dtype=bool)
    sample_area_mask[pixel_indices] = True
    return sample_area_mask.reshape(nrows, ncols)


def choose_ds_segments(ds_segments_bounds, centr_df, ppm):
    centr_segm_min_mz, centr_segm_max_mz = centr_df.mz.agg([np.min, np.max])
    centr_segm_min_mz -= centr_segm_min_mz * ppm * 1e-6
    centr_segm_max_mz += centr_segm_max_mz * ppm * 1e-6

    ds_segm_n = len(ds_segments_bounds)
    first_ds_segm_i = np.searchsorted(ds_segments_bounds[:, 0], centr_segm_min_mz, side='right') - 1
    first_ds_segm_i = max(0, first_ds_segm_i)
    last_ds_segm_i = np.searchsorted(ds_segments_bounds[:, 1], centr_segm_max_mz, side='left')  # last included
    last_ds_segm_i = min(ds_segm_n - 1, last_ds_segm_i)
    return first_ds_segm_i, last_ds_segm_i


def create_process_segment(ds_bucket, output_bucket, formula_images_prefix, ds_segments_bounds,
                           ds_segm_keys, coordinates, image_gen_config):
    sample_area_mask = make_sample_area_mask(coordinates)
    nrows, ncols = ds_dims(coordinates)
    compute_metrics = make_compute_image_metrics(sample_area_mask, nrows, ncols, image_gen_config)
    ppm = image_gen_config['ppm']

    def process_centr_segment(bucket, key, data_stream, ibm_cos):
        segm_i = f'{key.split("/")[-2]}/{key.split("/")[-1].split(".msgpack")[0]}'
        print(f'Reading centroids segment {segm_i} from {key}')
        centr_df = pd.read_msgpack(data_stream._raw_stream)

        first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segments_bounds, centr_df, ppm)
        print(f'Reading dataset segments {first_ds_segm_i}-{last_ds_segm_i}')
        # (ds_bucket, ds_segments_prefix, first_segm_i, last_segm_i, ibm_cos):
        sp_arr = read_ds_segments(ds_bucket, ds_segm_keys, first_ds_segm_i, last_ds_segm_i, ibm_cos)

        formula_images_it = gen_iso_images(sp_inds=sp_arr[:,0], sp_mzs=sp_arr[:,1], sp_ints=sp_arr[:,2],
                                           centr_df=centr_df,
                                           nrows=nrows, ncols=ncols, ppm=ppm, min_px=1)
        formula_metrics_df, formula_images = formula_image_metrics(formula_images_it, compute_metrics)

        print(f'Saving {len(formula_images)} images')
        ibm_cos.put_object(Bucket=output_bucket,
                           Key=f'{formula_images_prefix}/{segm_i}.pickle',
                           Body=pickle.dumps(formula_images))

        print(f'Segment {segm_i} finished')

        return formula_metrics_df

    return process_centr_segment
