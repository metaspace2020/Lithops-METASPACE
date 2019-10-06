import pickle
import pywren_ibm_cloud as pywren

from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd

from annotation_pipeline.check_results import get_reference_results, check_results, log_bad_results
from annotation_pipeline.fdr import build_fdr_rankings, calculate_fdrs
from annotation_pipeline.image import create_process_segment
from annotation_pipeline.segment import define_ds_segments, chunk_spectra, segment_spectra, segment_centroids,\
    clip_centroids_df_per_chunk, define_centr_segments
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

    def split_ds(self):
        clean_from_cos(self.config, self.config["storage"]["ds_bucket"], self.input_data["ds_chunks"])
        self.specra_chunks_keys = chunk_spectra(self.config, self.input_data, self.sp_n, self.imzml_parser, self.coordinates)

    def segment_ds(self):
        clean_from_cos(self.config, self.config["storage"]["ds_bucket"], self.input_data["ds_segments"])
        self.ds_segments_bounds = define_ds_segments(self.imzml_parser, self.ds_segm_size_mb, sample_ratio=0.05)
        self.ds_segm_n = len(self.ds_segments_bounds)
        self.ds_segm_keys = segment_spectra(self.config, self.config["storage"]["ds_bucket"],
                                            self.input_data["ds_chunks"], self.input_data["ds_segments"],
                                            self.ds_segments_bounds)
        logger.info(f'Segmented dataset chunks into {self.ds_segm_n} segments')

    def segment_centroids(self):
        mz_min, mz_max = self.ds_segments_bounds[0, 0], self.ds_segments_bounds[-1, 1]

        clean_from_cos(self.config, self.config["storage"]["db_bucket"], self.input_db["clipped_centroids_chunks"])
        self.centr_n = clip_centroids_df_per_chunk(self.config, self.config["storage"]["db_bucket"],
                                                   self.input_db["centroids_chunks"],
                                                   self.input_db["clipped_centroids_chunks"], mz_min, mz_max)

        clean_from_cos(self.config, self.config["storage"]["db_bucket"], self.input_db["centroids_segments"])
        self.centr_segm_lower_bounds = define_centr_segments(self.config, self.config["storage"]["db_bucket"],
                                                             self.input_db["clipped_centroids_chunks"], self.centr_n,
                                                             self.ds_segm_n, self.ds_segm_size_mb)
        self.centr_segm_n = len(self.centr_segm_lower_bounds)
        segment_centroids(self.config, self.config["storage"]["db_bucket"], self.input_db["clipped_centroids_chunks"],
                          self.input_db["centroids_segments"], self.centr_segm_lower_bounds)
        logger.info(f'Segmented centroids chunks into {self.centr_segm_n} segments')

    def annotate(self):
        logger.info('Annotating...')
        clean_from_cos(self.config, self.config["storage"]["output_bucket"], self.output["formula_images"])

        process_centr_segment = create_process_segment(self.config["storage"]["ds_bucket"],
                                                       self.config["storage"]["output_bucket"], self.output["formula_images"],
                                                       self.ds_segments_bounds, self.ds_segm_keys, self.coordinates,
                                                       self.image_gen_config)

        pw = pywren.ibm_cf_executor(config=self.config, runtime_memory=2048)
        futures = pw.map(process_centr_segment, f'{self.config["storage"]["db_bucket"]}/{self.input_db["centroids_segments"]}/')
        formula_metrics_list = pw.get_result(futures)
        append_pywren_stats(futures, pw.config['pywren']['runtime_memory'])

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
            obj = ibm_cos.get_object(Bucket=self.config['storage']['output_bucket'],
                                     Key=f'{self.output["formula_images"]}/{segm_i}.pickle')

            segm_images = pickle.loads(obj['Body'].read())
            images.update(segm_images)

        return dict((formula_i, images[formula_i]) for formula_i in self.results_df.index)

    def check_results(self):
        results_df = self.get_results()
        metaspace_options = self.config.get('metaspace_options', {})
        reference_results = get_reference_results(metaspace_options, self.input_data['metaspace_id'])

        checked_results = check_results(results_df, reference_results)

        log_bad_results(**checked_results)
        return checked_results

