import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import msgpack_numpy as msgpack

from annotation_pipeline.utils import ds_dims, get_pixel_indices
from annotation_pipeline.validate import make_compute_image_metrics, formula_image_metrics


def gen_iso_images(ds_segm_sp_array_it, centr_df, nrows, ncols, ppm=3):
    for sp_arr in ds_segm_sp_array_it:
        sp_inds = sp_arr[:, 0]
        sp_mzs = sp_arr[:, 1]
        sp_ints = sp_arr[:, 2]

        if sp_inds.size > 0:
            ds_segm_mz_min = sp_mzs[0] - sp_mzs[0] * ppm * 1e-6
            ds_segm_mz_max = sp_mzs[-1] + sp_mzs[-1] * ppm * 1e-6

            centr_df_slice = centr_df[
                (centr_df.mz >= ds_segm_mz_min) & (centr_df.mz <= ds_segm_mz_max)
            ]

            centr_mzs = centr_df_slice.mz.values
            centr_f_inds = centr_df_slice.formula_i.values
            centr_p_inds = centr_df_slice.peak_i.values
            centr_ints = centr_df_slice.int.values

            lower = centr_mzs - centr_mzs * ppm * 1e-6
            upper = centr_mzs + centr_mzs * ppm * 1e-6
            lower_inds = np.searchsorted(sp_mzs, lower, 'l')
            upper_inds = np.searchsorted(sp_mzs, upper, 'r')

            for i, (lo_i, up_i) in enumerate(zip(lower_inds, upper_inds)):
                m = None
                if up_i - lo_i > 0:
                    data = sp_ints[lo_i:up_i].copy()
                    inds = sp_inds[lo_i:up_i]
                    row_inds = np.uint16(inds / ncols, copy=True)
                    col_inds = np.uint16(inds % ncols, copy=True)
                    m = coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols))
                yield centr_f_inds[i], centr_p_inds[i], centr_ints[i], m


def read_ds_segment(ds_bucket, ds_segm_prefix, segm_i, ibm_cos):
    ds_segm_key = f'{ds_segm_prefix}/{segm_i}.msgpack'
    data_stream = ibm_cos.get_object(Bucket=ds_bucket, Key=ds_segm_key)['Body']
    data = msgpack.loads(data_stream.read())
    if type(data) == list:
        sp_arr = np.concatenate(data)
    else:
        sp_arr = data

    if len(sp_arr) > 0:
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


def create_process_segment(ds_bucket, output_bucket, ds_segm_prefix, formula_images_prefix, ds_segments_bounds,
                           coordinates, image_gen_config):
    sample_area_mask = make_sample_area_mask(coordinates)
    nrows, ncols = ds_dims(coordinates)
    compute_metrics = make_compute_image_metrics(sample_area_mask, nrows, ncols, image_gen_config)
    ppm = image_gen_config['ppm']

    def process_centr_segment(obj, id, ibm_cos):
        print(f'Reading centroids segment {obj.key}')
        # read database relevant part
        centr_df = pd.read_msgpack(obj.data_stream._raw_stream)
        # find range of datasets
        first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segments_bounds, centr_df, ppm)
        print(f'Reading dataset segments {first_ds_segm_i}-{last_ds_segm_i}')
        # read all segments on-demand in loop from COS
        ds_segm_sp_array_it = (
            read_ds_segment(ds_bucket, ds_segm_prefix, segm_i, ibm_cos)
            for segm_i in range(first_ds_segm_i, last_ds_segm_i+1)
        )

        formula_images_it = gen_iso_images(ds_segm_sp_array_it=ds_segm_sp_array_it,
                                           centr_df=centr_df,
                                           nrows=nrows, ncols=ncols, ppm=ppm)
        formula_metrics_df, formula_images = formula_image_metrics(formula_images_it, compute_metrics, min_px=1)

        print(f'Saving {len(formula_images)} images')
        ibm_cos.put_object(Bucket=output_bucket,
                           Key=f'{formula_images_prefix}/{id}.pickle',
                           Body=pickle.dumps(formula_images))

        print(f'Centroids segment {obj.key} finished')

        return formula_metrics_df

    return process_centr_segment
