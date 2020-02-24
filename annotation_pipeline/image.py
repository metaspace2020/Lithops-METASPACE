import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack

from annotation_pipeline.utils import ds_dims, get_pixel_indices, read_object_with_retry
from annotation_pipeline.validate import make_compute_image_metrics, formula_image_metrics
from annotation_pipeline.segment import ISOTOPIC_PEAK_N


class ImagesManager:
    min_memory_allowed = 128 * 1024 ** 2  # 128MB

    def __init__(self, internal_storage, bucket, max_formula_images_size):
        if max_formula_images_size < self.__class__.min_memory_allowed:
            raise Exception(f'There isn\'t enough memory to generate images, consider increasing PyWren\'s memory.')

        self.formula_metrics = {}
        self.formula_images = {}
        self.cloud_objs = []

        self._formula_images_size = 0
        self._max_formula_images_size = max_formula_images_size
        self._internal_storage = internal_storage
        self._bucket = bucket
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
        if self._formula_images_size > self._max_formula_images_size:
            self.save_images()
            self.formula_images.clear()
            self._formula_images_size = 0

    def add_f_metrics(self, f_i, f_metrics):
        self.formula_metrics[f_i] = f_metrics

    def save_images(self):
        if self.formula_images:
            print(f'Saving {len(self.formula_images)} images')
            cloud_obj = self._internal_storage.put_object(pickle.dumps(self.formula_images), bucket=self._bucket)
            self.cloud_objs.append(cloud_obj)
            self._partition += 1
        else:
            print(f'No images to save')

    def finish(self):
        self.save_images()
        self.formula_images.clear()
        self._formula_images_size = 0
        return self.cloud_objs


def gen_iso_images(sp_inds, sp_mzs, sp_ints, centr_df, nrows, ncols, ppm=3, min_px=1):
    # assume sp data is sorted by mz order ascending
    # assume centr data is sorted by mz order ascending

    centr_f_inds = centr_df.formula_i.values
    centr_p_inds = centr_df.peak_i.values
    centr_mzs = centr_df.mz.values
    centr_ints = centr_df.int.values

    def yield_buffer(buffer):
        while len(buffer) < ISOTOPIC_PEAK_N:
            buffer.append((buffer[0][0], len(buffer) - 1, 0, None))
        buffer = np.array(buffer)
        buffer = buffer[buffer[:, 1].argsort()]  # sort order by peak ascending
        buffer = pd.DataFrame(buffer, columns=['formula_i', 'peak_i', 'centr_ints', 'image'])
        buffer.sort_values('peak_i', inplace=True)
        return buffer.formula_i[0], buffer.centr_ints, buffer.image

    if len(sp_inds) > 0:
        lower = centr_mzs - centr_mzs * ppm * 1e-6
        upper = centr_mzs + centr_mzs * ppm * 1e-6
        lower_idx = np.searchsorted(sp_mzs, lower, 'l')
        upper_idx = np.searchsorted(sp_mzs, upper, 'r')
        ranges_df = pd.DataFrame({'formula_i': centr_f_inds, 'lower_idx': lower_idx, 'upper_idx': upper_idx}).sort_values('formula_i')

        buffer = []
        for df_index, df_row in ranges_df.iterrows():
            if len(buffer) != 0 and buffer[0][0] != centr_f_inds[df_index]:
                yield yield_buffer(buffer)
                buffer = []

            l, u = df_row['lower_idx'], df_row['upper_idx']
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
        data = read_object_with_retry(ibm_cos, ds_bucket, ds_segm_key, msgpack.load)

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


def create_process_segment(ds_bucket, output_bucket, ds_segm_prefix, ds_segments_bounds,
                           coordinates, image_gen_config, pw_mem_mb, ds_segm_size_mb):
    sample_area_mask = make_sample_area_mask(coordinates)
    nrows, ncols = ds_dims(coordinates)
    compute_metrics = make_compute_image_metrics(sample_area_mask, nrows, ncols, image_gen_config)
    ppm = image_gen_config['ppm']

    def process_centr_segment(obj, ibm_cos, internal_storage):
        print(f'Reading centroids segment {obj.key}')
        # read database relevant part
        try:
            centr_df = pd.read_msgpack(obj.data_stream)
        except:
            centr_df = read_object_with_retry(ibm_cos, obj.bucket, obj.key, pd.read_msgpack)

        # find range of datasets
        first_ds_segm_i, last_ds_segm_i = choose_ds_segments(ds_segments_bounds, centr_df, ppm)
        print(f'Reading dataset segments {first_ds_segm_i}-{last_ds_segm_i}')
        # read all segments in loop from COS
        sp_arr = read_ds_segments(ds_bucket, ds_segm_prefix, first_ds_segm_i, last_ds_segm_i, ibm_cos)

        formula_images_it = gen_iso_images(sp_inds=sp_arr[:,0], sp_mzs=sp_arr[:,1], sp_ints=sp_arr[:,2],
                                           centr_df=centr_df, nrows=nrows, ncols=ncols, ppm=ppm, min_px=1)
        safe_mb = 1024
        max_formula_images_mb = (pw_mem_mb - safe_mb - (last_ds_segm_i - first_ds_segm_i + 1) * ds_segm_size_mb) // 2
        print(f'max_formula_images_mb: {max_formula_images_mb}')
        images_manager = ImagesManager(internal_storage, output_bucket, max_formula_images_mb * 1024 ** 2)
        formula_image_metrics(formula_images_it, compute_metrics, images_manager)
        images_cloud_objs = images_manager.finish()

        print(f'Centroids segment {obj.key} finished')
        formula_metrics_df = pd.DataFrame.from_dict(images_manager.formula_metrics, orient='index')
        formula_metrics_df.index.name = 'formula_i'
        return formula_metrics_df, images_cloud_objs

    return process_centr_segment
