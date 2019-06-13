import json
import pickle
from pathlib import Path
import pywren_ibm_cloud as pywren

from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd

from annotation_pipeline.fdr import build_fdr_rankings, calculate_fdrs
from annotation_pipeline.image import create_process_segment
from annotation_pipeline.segment import define_ds_segments, chunk_spectra, segment_spectra
from annotation_pipeline.segment import clip_centroids_df, calculate_centroids_segments_n, segment_centroids
from annotation_pipeline.utils import ds_imzml_path, clean_from_cos, get_ibm_cos_client, append_pywren_stats
from annotation_pipeline.utils import logger


class Pipeline(object):

    def __init__(self, config, input_config):
        self.config = config
        self.storage = config['storage']
        self.input_data = input_config['dataset']
        self.input_db = input_config['molecular_db']
        self.output = input_config['output']

        self.ds_segm_size_mb = 5
        self.image_gen_config = {
            "q": 99,
            "do_preprocessing": False,
            "nlevels": 30,
            "ppm": 3.0
        }

    def __call__(self, *args, **kwargs):
        self.load_ds()
        self.split_ds()
        self.segment_ds()
        self.segment_centroids()
        self.annotate()
        self.run_fdr()

    def load_ds(self):
        self.imzml_parser = ImzMLParser(ds_imzml_path(self.input_data['path']))
        self.coordinates = [coo[:2] for coo in self.imzml_parser.coordinates]
        self.sp_n = len(self.coordinates)
        logger.info(f'Parsed imzml: {self.sp_n} spectra found')

        ds_config = json.load(open(Path(self.input_data['path']) / 'config.json'))
        self.isotope_gen_config = ds_config['isotope_generation']

    def split_ds(self):
        clean_from_cos(self.config, self.config["storage"]["ds_bucket"], self.input_data["ds_chunks"])
        self.specra_chunks_keys = chunk_spectra(self.config, self.input_data, self.sp_n, self.imzml_parser, self.coordinates)

    def segment_ds(self):
        clean_from_cos(self.config, self.config["storage"]["ds_bucket"], self.input_data["ds_segments"])
        self.ds_segments_bounds = define_ds_segments(self.imzml_parser, self.ds_segm_size_mb, sample_ratio=0.05)
        self.ds_segm_n = len(self.ds_segments_bounds)
        segment_spectra(self.config, self.config["storage"]["ds_bucket"], self.input_data["ds_chunks"],
                        self.input_data["ds_segments"], self.ds_segments_bounds)
        logger.info(f'Segmented dataset chunks into {self.ds_segm_n} segments')

    def segment_centroids(self):
        clean_from_cos(self.config, self.config["storage"]["db_bucket"], self.input_db["centroids_segments"])
        pw = pywren.ibm_cf_executor(config=self.config, runtime_memory=2048)

        db_bucket_key = f'{self.config["storage"]["db_bucket"]}/{self.input_db["centroids_pandas"]}'
        mz_min, mz_max = self.ds_segments_bounds[0, 0], self.ds_segments_bounds[-1, 1]
        futures = pw.map(clip_centroids_df, [[db_bucket_key, mz_min, mz_max]])
        self.centr_n = pw.get_result(futures)
        append_pywren_stats(clip_centroids_df.__name__, 2048, futures)
        logger.info(f'Prepared {self.centr_n} centroids')

        self.centr_segm_n = calculate_centroids_segments_n(self.centr_n, self.ds_segm_n, self.ds_segm_size_mb)
        futures = pw.map(segment_centroids, [[self.config["storage"]["ds_bucket"], self.centr_segm_n, self.input_db["centroids_segments"]]])
        pw.get_result(futures)
        append_pywren_stats(segment_centroids.__name__, 2048, futures)
        logger.info(f'Segmented centroids into {self.centr_segm_n} segments')

    def annotate(self):
        logger.info('Annotating...')
        clean_from_cos(self.config, self.config["storage"]["output_bucket"], self.output["formula_images"])

        process_centr_segment = create_process_segment(self.config["storage"]["ds_bucket"], self.input_data["ds_segments"],
                                                       self.config["storage"]["output_bucket"], self.output["formula_images"],
                                                       self.ds_segments_bounds, self.coordinates, self.image_gen_config)

        pw = pywren.ibm_cf_executor(config=self.config, runtime_memory=2048)
        futures = pw.map(process_centr_segment, f'{self.config["storage"]["db_bucket"]}/{self.input_db["centroids_segments"]}')
        formula_metrics_list = pw.get_result(futures)
        append_pywren_stats(process_centr_segment.__name__, 2048, futures)

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
        ibm_cos = get_ibm_cos_client(self.config)
        for segm_i in range(self.centr_segm_n):
            logger.info(f'Downloading pickled images #{segm_i}')
            obj = ibm_cos.get_object(Bucket=self.config['storage']['output_bucket'],
                                     Key=f'{self.output["formula_images"]}/{segm_i}.pickle')

            segm_images = pickle.loads(obj['Body'].read())
            images.update(segm_images)

        return dict((formula_i, images[formula_i]) for formula_i in self.results_df.index)
