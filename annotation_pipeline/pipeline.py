import logging

from .dataset_segmentation import generate_segm_intervals, split_spectra_into_segments, clean_segments
from .annotation import annotate_spectra, merge_annotation_results

logger = logging.getLogger(name='annotation_pipeline')


def annotate_dataset(config, input_data, input_db, segm_n=256):

    logger.info('Generating segments')
    segm_intervals = generate_segm_intervals(config, input_db, segm_n)
    split_spectra_into_segments(config, input_data, segm_n, segm_intervals)

    logger.info('Annotating')
    results = annotate_spectra(config, input_data, input_db, segm_n)
    formula_scores_df, formula_images = merge_annotation_results(results)

    logger.info('Cleaning up')
    clean_segments(config, input_data)

    return formula_scores_df, formula_images
