import json
import pickle
from pathlib import Path
from shutil import rmtree

from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd

from annotation_pipeline_v2.fdr import build_fdr_rankings, calculate_fdrs
from annotation_pipeline_v2.image import create_process_segment
from annotation_pipeline_v2.segment import define_ds_segments, segment_spectra
from annotation_pipeline_v2.segment import clip_centroids_df, calculate_centroids_segments_n, segment_centroids
from annotation_pipeline_v2.utils import ds_imzml_path
from annotation_pipeline_v2.utils import logger


class Pipeline(object):

    def __init__(self, config, input_config):
        self.config = config
        self.input_data = input_config['dataset']
        self.input_db = input_config['molecular_db']
        self.ds_segments_path = Path(self.input_data['ds_segments'])
        self.centr_segments_path = Path(self.input_data['centr_segments'])
        self.formula_images_path = Path(self.input_data['formula_images'])

        self.ds_segm_size_mb = 5
        self.image_gen_config = {
            "q": 99,
            "do_preprocessing": False,
            "nlevels": 30,
            "ppm": 3.0
        }

    def load_ds(self):
        self.imzml_parser = ImzMLParser(ds_imzml_path(self.input_data['path']))
        self.coordinates = [coo[:2] for coo in self.imzml_parser.coordinates]
        self.sp_n = len(self.coordinates)
        logger.info(f'Parsed imzml: {self.sp_n} spectra found')

        ds_config = json.load(open(Path(self.input_data['path']) / 'config.json'))
        self.isotope_gen_config = ds_config['isotope_generation']

    def segment_ds(self):
        self.ds_segments = define_ds_segments(self.imzml_parser, self.ds_segm_size_mb, sample_ratio=0.05)
        segment_spectra(self.imzml_parser, self.coordinates, self.ds_segments, self.ds_segments_path)

    def segment_centroids(self):
        centroids_df = pd.read_pickle(self.input_db['centroids_pandas']).sort_values('mz')
        centroids_df = centroids_df[centroids_df.mz > 0]

        mz_min, mz_max = self.ds_segments[0, 0], self.ds_segments[-1, 1]
        self.centr_df = clip_centroids_df(centroids_df, mz_min, mz_max)
        logger.info(f'Prepared {self.centr_df.shape[0]} centroids')

        self.centr_segm_n = calculate_centroids_segments_n(self.centr_df, self.ds_segments, self.ds_segm_size_mb)
        segment_centroids(self.centr_df, self.centr_segm_n, self.centr_segments_path)

    def annotate(self):
        logger.info(f'Annotating...')
        rmtree(self.formula_images_path, ignore_errors=True)
        self.formula_images_path.mkdir(parents=True)

        process_centr_segment = create_process_segment(self.ds_segments, self.coordinates, self.image_gen_config,
                                                       self.ds_segments_path, self.centr_segments_path,
                                                       self.formula_images_path)

        formula_metrics_list = [process_centr_segment(i) for i in range(self.centr_segm_n)]

        self.formula_metrics_df = pd.concat(formula_metrics_list)
        logger.info(f'Metrics calculated: {self.formula_metrics_df.shape[0]}')

    def run_fdr(self):
        self.rankings_df = build_fdr_rankings(self.config, self.input_data, self.input_db, self.formula_metrics_df)
        self.fdrs = calculate_fdrs(self.config, self.input_data, self.rankings_df)

        logger.info(f'Number of annotations at with FDR less than:')
        for fdr_step in [0.05, 0.1, 0.2, 0.5]:
            logger.info(f'{fdr_step*100:2.0f}%: {(self.fdrs.fdr < fdr_step).sum()}')

    def get_results(self):
        results_df = self.formula_metrics_df.merge(self.fdrs, left_index=True, right_index=True)
        results_df = results_df[~results_df.adduct.isna()]
        results_df = results_df.sort_values('fdr')
        self.results_df = results_df
        return results_df

    def get_images(self):
        images = {}
        for segm_i in range(self.centr_segm_n):
            segm_images = pickle.load(open(self.formula_images_path / f'images_{segm_i}.pickle', 'rb'))
            images.update(segm_images)

        return dict((formula_i, images[formula_i]) for formula_i in self.results_df.index)
