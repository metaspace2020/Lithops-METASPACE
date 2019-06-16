import logging

from .dataset import read_dataset_coords, real_pixel_indices
from .molecular_db import store_centroids_database
from .dataset_segmentation import generate_segm_intervals, split_spectra_into_segments, clean_segments
from .annotation import annotate_spectra, merge_annotation_results

logger = logging.getLogger(name='annotation_pipeline')


def annotate_dataset(config, input_data, input_db, segm_n=256):

    logger.info('Storing centroids database')
    store_centroids_database(config, input_db)

    spectra_coords = read_dataset_coords(config, input_data)
    pixel_indices, nrows, ncols = real_pixel_indices(spectra_coords)

    logger.info('Generating segments')
    segm_intervals = generate_segm_intervals(config, input_db, segm_n)
    split_spectra_into_segments(config, input_data, segm_n, segm_intervals)

    logger.info('Annotating')
    results = annotate_spectra(config, input_data, input_db, segm_n, pixel_indices, nrows, ncols)
    formula_scores_df, formula_images = merge_annotation_results(results)

    logger.info('Cleaning up')
    clean_segments(config, input_data)

    return formula_scores_df, formula_images
