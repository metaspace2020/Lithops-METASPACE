import pickle
from itertools import chain
import pywren_ibm_cloud as pywren
import pandas as pd

from annotation_pipeline.check_results import get_reference_results, check_results, log_bad_results
from annotation_pipeline.fdr import build_fdr_rankings, calculate_fdrs
from annotation_pipeline.image import create_process_segment
from annotation_pipeline.segment import define_ds_segments, chunk_spectra, segment_spectra, segment_centroids, \
    clip_centr_df, define_centr_segments, get_imzml_reader
from annotation_pipeline.cache import PipelineCacher
from annotation_pipeline.utils import append_pywren_stats, logger, read_cloud_object_with_retry


class Pipeline(object):

    def __init__(self, config, input_config, use_cache=True):
        self.config = config
        self.storage = config['storage']
        self.input_config_ds = input_config['dataset']
        self.input_config_db = input_config['molecular_db']
        self.use_cache = use_cache
        self.pywren_executor = pywren.function_executor(config=self.config, runtime_memory=2048)

        self.cacher = PipelineCacher(self.pywren_executor, self.input_config_ds["name"])
        if not self.use_cache:
            self.cacher.clean()

        self.ds_segm_size_mb = 100
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
        imzml_cache_key = f'{self.cacher.prefix}/load_ds.cache'

        if self.cacher.exists(imzml_cache_key):
            self.imzml_reader, self.imzml_cobject = self.cacher.load(imzml_cache_key)
            logger.info(f'Loaded imzml from cache, {len(self.imzml_reader.coordinates)} spectra found')
        else:
            self.imzml_reader, self.imzml_cobject = get_imzml_reader(self.pywren_executor, self.input_config_ds['imzml_path'])
            logger.info(f'Parsed imzml: {len(self.imzml_reader.coordinates)} spectra found')
            self.cacher.save((self.imzml_reader, self.imzml_cobject), imzml_cache_key)

    def split_ds(self):
        ds_chunks_cache_key = f'{self.cacher.prefix}/split_ds.cache'

        if self.cacher.exists(ds_chunks_cache_key):
            self.ds_chunks_cobjects = self.cacher.load(ds_chunks_cache_key)
            logger.info(f'Loaded {len(self.ds_chunks_cobjects)} dataset chunks from cache')
        else:
            self.ds_chunks_cobjects = chunk_spectra(self.pywren_executor, self.input_config_ds['ibd_path'],
                                                    self.imzml_cobject, self.imzml_reader)
            logger.info(f'Uploaded {len(self.ds_chunks_cobjects)} dataset chunks')
            self.cacher.save(self.ds_chunks_cobjects, ds_chunks_cache_key)

    def segment_ds(self):
        ds_segments_cache_key = f'{self.cacher.prefix}/segment_ds.cache'

        if self.cacher.exists(ds_segments_cache_key):
            self.ds_segments_bounds, self.ds_segms_cobjects, self.ds_segms_len = \
                self.cacher.load(ds_segments_cache_key)
            logger.info(f'Loaded {len(self.ds_segms_cobjects)} dataset segments from cache')
        else:
            sample_sp_n = 1000
            self.ds_segments_bounds = define_ds_segments(self.pywren_executor, self.input_config_ds["ibd_path"],
                                                         self.imzml_cobject, self.ds_segm_size_mb, sample_sp_n)
            self.ds_segms_cobjects, self.ds_segms_len = \
                segment_spectra(self.pywren_executor, self.ds_chunks_cobjects, self.ds_segments_bounds, self.ds_segm_size_mb)
            logger.info(f'Segmented dataset chunks into {len(self.ds_segms_cobjects)} segments')
            self.cacher.save((self.ds_segments_bounds, self.ds_segms_cobjects, self.ds_segms_len), ds_segments_cache_key)

        self.ds_segm_n = len(self.ds_segms_cobjects)

    def segment_centroids(self):
        mz_min, mz_max = self.ds_segments_bounds[0, 0], self.ds_segments_bounds[-1, 1]
        db_segments_cache_key = f'{self.cacher.prefix}/segment_centroids.cache'

        if self.cacher.exists(db_segments_cache_key):
            self.clip_centr_chunks_cobjects, self.db_segms_cobjects = self.cacher.load(db_segments_cache_key)
            logger.info(f'Loaded {len(self.db_segms_cobjects)} centroids segments from cache')
        else:
            self.clip_centr_chunks_cobjects, centr_n = \
                clip_centr_df(self.pywren_executor, self.config["storage"]["db_bucket"],
                              self.input_config_db["centroids_chunks"], mz_min, mz_max)
            centr_segm_lower_bounds = define_centr_segments(self.pywren_executor, self.clip_centr_chunks_cobjects,
                                                                 centr_n, self.ds_segm_n, self.ds_segm_size_mb)
            self.db_segms_cobjects = segment_centroids(self.pywren_executor, self.clip_centr_chunks_cobjects,
                                                       centr_segm_lower_bounds)
            logger.info(f'Segmented centroids chunks into {len(self.db_segms_cobjects)} segments')
            self.cacher.save((self.clip_centr_chunks_cobjects, self.db_segms_cobjects), db_segments_cache_key)

        self.centr_segm_n = len(self.db_segms_cobjects)

    def annotate(self):
        annotations_cache_key = f'{self.cacher.prefix}/annotate.cache'

        if self.cacher.exists(annotations_cache_key):
            self.formula_metrics_df, self.images_cloud_objs = self.cacher.load(annotations_cache_key)
            logger.info(f'Loaded {self.formula_metrics_df.shape[0]} metrics from cache')
        else:
            logger.info('Annotating...')
            if self.ds_segm_n * self.ds_segm_size_mb > 5000:
                memory_capacity_mb = 4096
            else:
                memory_capacity_mb = 2048
            process_centr_segment = create_process_segment(self.ds_segms_cobjects,
                                                           self.ds_segments_bounds, self.ds_segms_len, self.imzml_reader,
                                                           self.image_gen_config, memory_capacity_mb, self.ds_segm_size_mb)

            futures = self.pywren_executor.map(process_centr_segment, self.db_segms_cobjects, runtime_memory=memory_capacity_mb)
            formula_metrics_list, images_cloud_objs = zip(*self.pywren_executor.get_result(futures))
            self.formula_metrics_df = pd.concat(formula_metrics_list)
            self.images_cloud_objs = list(chain(*images_cloud_objs))
            append_pywren_stats(futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(self.images_cloud_objs))
            logger.info(f'Metrics calculated: {self.formula_metrics_df.shape[0]}')
            self.cacher.save((self.formula_metrics_df, self.images_cloud_objs), annotations_cache_key)

    def run_fdr(self):
        fdrs_cache_key = f'{self.cacher.prefix}/run_fdr.cache'

        if self.cacher.exists(fdrs_cache_key):
            self.rankings_df, self.fdrs = self.cacher.load(fdrs_cache_key)
            logger.info('Loaded fdrs from cache')
        else:
            self.rankings_df = build_fdr_rankings(self.pywren_executor, self.config["storage"]["db_bucket"],
                                                  self.input_config_ds, self.input_config_db, self.formula_metrics_df)
            self.fdrs = calculate_fdrs(self.pywren_executor, self.rankings_df)
            self.cacher.save((self.rankings_df, self.fdrs), fdrs_cache_key)

        logger.info('Number of annotations at with FDR less than:')
        for fdr_step in [0.05, 0.1, 0.2, 0.5]:
            logger.info(f'{fdr_step*100:2.0f}%: {(self.fdrs.fdr < fdr_step).sum()}')

    def get_results(self):
        results_df = self.formula_metrics_df.merge(self.fdrs, left_index=True, right_index=True)
        results_df = results_df[~results_df.adduct.isna()]
        results_df = results_df.sort_values('fdr')
        self.results_df = results_df
        return results_df

    def get_images(self):
        # Only download interesting images, to prevent running out of memory
        targets = set(self.results_df.index[self.results_df.fdr <= 0.5])

        def get_target_images(images_cobject, storage):
            images = {}
            segm_images = pickle.loads(read_cloud_object_with_retry(storage, images_cobject))
            for k, v in segm_images.items():
                if k in targets:
                    images[k] = v
            return images

        futures = self.pywren_executor.map(get_target_images, self.images_cloud_objs, runtime_memory=1024)
        all_images = {}
        for image_set in self.pywren_executor.get_result(futures):
            all_images.update(image_set)

        return all_images

    def check_results(self):
        results_df = self.get_results()
        metaspace_options = self.config.get('metaspace_options', {})
        reference_results = get_reference_results(metaspace_options, self.input_config_ds['metaspace_id'])

        checked_results = check_results(results_df, reference_results)

        log_bad_results(**checked_results)
        return checked_results

    def clean(self):
        self.cacher.clean()
