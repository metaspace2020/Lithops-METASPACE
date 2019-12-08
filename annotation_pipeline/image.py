import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack

from annotation_pipeline.utils import ds_dims, get_pixel_indices
from annotation_pipeline.validate import make_compute_image_metrics, formula_image_metrics
from annotation_pipeline.segment import ISOTOPIC_PEAK_N


class ImagesManager:
    max_formula_images_size = 512 * 1024 ** 2  # 512MB

    def __init__(self, ibm_cos, bucket, prefix):
        self.formula_metrics = {}
        self.formula_images = {}

        self._formula_images_size = 0
        self._ibm_cos = ibm_cos
        self._bucket = bucket
        self._prefix = prefix
        self._partition = 0

    def __call__(self, f_i, f_metrics, f_images):
        self.add_f_metrics(f_i, f_metrics)
        self.add_f_images(f_i, f_images)

    @staticmethod
    def images_size(f_images):
        return sum(img.data.nbytes + img.row.nbytes + img.col.nbytes for img in f_images if img is not None)

    def add_f_images(self, f_i, f_images):
        self.formula_images[f_i] = f_images
        self._formula_images_size += ImagesManager.images_size(f_images)
        if self._formula_images_size > self.__class__.max_formula_images_size:
            self.save_images()
            self.formula_images.clear()
            self._formula_images_size = 0

    def add_f_metrics(self, f_i, f_metrics):
        self.formula_metrics[f_i] = f_metrics

    def save_images(self):
        if self.formula_images:
            print(f'Saving {len(self.formula_images)} images')
            self._ibm_cos.put_object(Bucket=self._bucket,
                                     Key=f'{self._prefix}/{self._partition}.pickle',
                                     Body=pickle.dumps(self.formula_images))
            self._partition += 1
        else:
            print(f'No images to save')

    def finish(self):
        self.save_images()
        self.formula_images.clear()
        self._formula_images_size = 0


def gen_iso_images(sp_inds, sp_mzs, sp_ints, centr_f_inds, centr_p_inds, centr_mzs, centr_ints, nrows, ncols, ppm=3, min_px=1):
    # assume sp data is sorted by mz order ascending
    # assume centr data is sorted by mz order ascending

    def yield_buffer(buffer):
        while len(buffer) < ISOTOPIC_PEAK_N:
            buffer.append((buffer[0][0], len(buffer) - 1, 0, None))
        buffer = np.array(buffer)
        buffer = buffer[buffer[:, 1].argsort()]  # sort order by peak ascending
        f_i = buffer[0][0]
        f_ints = buffer[:, 2]
        f_images = buffer[:, 3]
        return f_i, f_ints, f_images

    if len(sp_inds) > 0:
        lower = centr_mzs - centr_mzs * ppm * 1e-6
        upper = centr_mzs + centr_mzs * ppm * 1e-6
        lower_idx = np.searchsorted(sp_mzs, lower, 'l')
        upper_idx = np.searchsorted(sp_mzs, upper, 'r')
        ranges_df = pd.DataFrame({'formula_i': centr_f_inds, 'range': zip(lower_idx, upper_idx)}).sort_values('formula_i')

        buffer = []
        for df_index, df_row in ranges_df.iterrows():
            if len(buffer) != 0 and buffer[0][0] != centr_f_inds[df_index]:
                yield yield_buffer(buffer)
                buffer = []

            l, u = df_row['range']
            m = None
            if u - l >= min_px:
                data = sp_ints[l:u]
                inds = sp_inds[l:u]
                row_inds = inds / ncols
                col_inds = inds % ncols
                m = coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols), copy=True)
            buffer.append((centr_f_inds[df_index], centr_p_inds[df_index], centr_ints[df_index], m))

        if len(buffer) != 0:
            yield yield_buffer(buffer)


def read_ds_segments(ds_bucket, ds_segm_prefix, first_segm_i, last_segm_i, ibm_cos):

    def read_ds_segment(ds_segm_key):
        data_stream = ibm_cos.get_object(Bucket=ds_bucket, Key=ds_segm_key)['Body']
        data = msgpack.loads(data_stream.read())
        if type(data) == list:
            sp_arr = np.concatenate(data)
        else:
            sp_arr = data
        return sp_arr

    with ThreadPoolExecutor(max_workers=128) as pool:
        ds_segm_keys = [f'{ds_segm_prefix}/{segm_i}.msgpack' for segm_i in range(first_segm_i, last_segm_i + 1)]
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


def create_process_segment(ds_bucket, output_bucket, ds_segm_prefix, formula_images_prefix, ds_segments_bounds,
                           coordinates, image_gen_config):
    sample_area_mask = make_sample_area_mask(coordinates)
    nrows, ncols = ds_dims(coordinates)
    compute_metrics = make_compute_image_metrics(sample_area_mask, nrows, ncols, image_gen_config)
    ppm = image_gen_config['ppm']

    def process_centr_segment(obj, id, ibm_cos):
        print(f'Reading centroids segment {obj.key}')
        centr_df = pd.read_msgpack(obj.data_stream._raw_stream)

        first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segments_bounds, centr_df, ppm)
        print(f'Reading dataset segments {first_ds_segm_i}-{last_ds_segm_i}')
        sp_arr = read_ds_segments(ds_bucket, ds_segm_prefix, first_ds_segm_i, last_ds_segm_i, ibm_cos)

        formula_images_it = gen_iso_images(sp_inds=sp_arr[:,0], sp_mzs=sp_arr[:,1], sp_ints=sp_arr[:,2],
                                           centr_f_inds=centr_df.formula_i.values, centr_p_inds=centr_df.peak_i.values,
                                           centr_mzs=centr_df.mz.values, centr_ints=centr_df.int.values,
                                           nrows=nrows, ncols=ncols, ppm=ppm, min_px=1)
        images_manager = ImagesManager(ibm_cos, output_bucket, f'{formula_images_prefix}/{id}')
        formula_image_metrics(formula_images_it, compute_metrics, images_manager)
        images_manager.finish()

        print(f'Centroids segment {obj.key} finished')
        formula_metrics_df = pd.DataFrame.from_dict(images_manager.formula_metrics, orient='index')
        formula_metrics_df.index.name = 'formula_i'
        return formula_metrics_df

    return process_centr_segment
