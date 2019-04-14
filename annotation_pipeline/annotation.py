import os

from scipy.sparse import coo_matrix
from collections import defaultdict
from pyImagingMSpec.image_measures import isotope_image_correlation, isotope_pattern_match
from cpyImagingMSpec import measure_of_chaos
import pywren_ibm_cloud as pywren
from itertools import chain
import numpy as np
import pandas as pd
import pickle


def sp_df_gen(sp_it, pixel_indices):
    for sp_id, mzs, intensities in sp_it:
        for mz, ints in zip(mzs, intensities):
            yield pixel_indices[sp_id], mz, ints


def gen_iso_images(spectra_it, pixel_indices, centr_df, nrows, ncols, ppm, min_px=1):
    if len(centr_df) > 0:
        # a bit slower than using pure numpy arrays but much shorter
        # may leak memory because of https://github.com/pydata/pandas/issues/2659 or smth else
        sp_df = pd.DataFrame(sp_df_gen(spectra_it, pixel_indices),
                             columns=['idx', 'mz', 'ints']).sort_values(by='mz')

        # -1, + 1 are needed to extend sf_peak_mz range so that it covers 100% of spectra
        centr_df = centr_df[(centr_df.mz >= sp_df.mz.min() - 1) &
                            (centr_df.mz <= sp_df.mz.max() + 1)]
        lower = centr_df.mz.map(lambda mz: mz - mz * ppm * 1e-6)
        upper = centr_df.mz.map(lambda mz: mz + mz * ppm * 1e-6)
        lower_idx = np.searchsorted(sp_df.mz, lower, 'l')
        upper_idx = np.searchsorted(sp_df.mz, upper, 'r')

        for i, (l, u) in enumerate(zip(lower_idx, upper_idx)):
            if u - l >= min_px:
                data = sp_df.ints[l:u].values
                if data.shape[0] > 0:
                    idx = sp_df.idx[l:u].values
                    row_inds = idx / ncols
                    col_inds = idx % ncols
                    m = coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols), copy=True)
                    yield centr_df.index[i], (centr_df.peak_i.iloc[i], m)


def merge_formula_images(iso_images):
    formula_images = defaultdict(list)
    for formula_i, (peak_i, img) in iso_images:
        formula_images[formula_i].append((peak_i, img))

    def filter_formula_images():
        filtered_f_images = {}
        for f_i, images in formula_images.items():
            images = sorted(images, key=lambda x: x[0])
            if len(images) > 1 and images[0][0] == 0:
                filtered_f_images[f_i] = images
        return filtered_f_images

    return filter_formula_images()


def score_formula_images(f_images, centroids_df, empty_image):
    formula_scores = []
    for formula_i, images in f_images.items():
        centr_ints = centroids_df.loc[formula_i].int.values

        image_list = [empty_image] * len(centr_ints)
        for peak_i, img in images:
            image_list[peak_i] = img.toarray()
        flat_image_list = [img.flat[:] for img in image_list]

        m1 = isotope_pattern_match(flat_image_list, centr_ints)
        m2 = isotope_image_correlation(flat_image_list, centr_ints[1:])
        m3 = measure_of_chaos(image_list[0], nlevels=30)
        formula_scores.append([formula_i, m1, m2, m3])

    formula_scores_df = pd.DataFrame(formula_scores,
                                     columns=['formula_i', 'm1', 'm2', 'm3']).set_index('formula_i')
    formula_scores_df['msm'] = formula_scores_df.m1 * formula_scores_df.m2 * formula_scores_df.m3
    return formula_scores_df


def filter_formula_images(formula_images, formula_scores_df):
    return {f_i: images
            for (f_i, images) in formula_images.items()
            if f_i in formula_scores_df.index}


def annotate_spectra(config, input_data, input_db, segm_n, pixel_indices, nrows, ncols):
    def annotate_segm_spectra(key, data_stream, ibm_cos):
        spectra = pickle.loads(data_stream.read())

        if not os.path.isfile('/tmp/centroids.pickle'):
            print("Read centroids DB from IBM COS")
            ibm_cos.download_file(input_db["bucket"], input_db['centroids_pandas'], '/tmp/centroids.pickle')

        with open('/tmp/centroids.pickle', 'rb') as centroids:
            centroids_df = pickle.load(centroids)

        iso_images = gen_iso_images(spectra, pixel_indices, centroids_df, nrows, ncols, ppm=3)
        formula_images = merge_formula_images(list(iso_images))
        formula_scores_df = score_formula_images(formula_images, centroids_df, empty_image)
        formula_scores_df = formula_scores_df[formula_scores_df.msm > 0]
        formula_images = filter_formula_images(formula_images, formula_scores_df)
        return formula_scores_df, formula_images

    empty_image = np.zeros((nrows, ncols))

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=1024)
    iterdata = [f'{input_data["bucket"]}/{input_data["segments"]}/{segm_i}.pickle' for segm_i in range(segm_n)]
    pw.map(annotate_segm_spectra, iterdata)
    results = pw.get_result()
    pw.clean()

    return results


def merge_annotation_results(results):
    formula_scores_list, formula_images_list = list(zip(*results))
    formula_scores_df = pd.concat(formula_scores_list)
    formula_images = dict(chain(*[segm_formula_images.items()
                                  for segm_formula_images in formula_images_list]))

    return formula_scores_df, formula_images
