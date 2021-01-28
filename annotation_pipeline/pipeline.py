from itertools import chain
from pathlib import Path

import lithops
import pandas as pd

from annotation_pipeline.check_results import get_reference_results, check_results, log_bad_results
from annotation_pipeline.fdr import build_fdr_rankings, calculate_fdrs, calculate_fdrs_vm
from annotation_pipeline.image import create_process_segment, make_sample_area_mask, get_target_images
from annotation_pipeline.molecular_db import upload_mol_dbs_from_dir, build_database, calculate_centroids, \
    validate_formula_cobjects, validate_peaks_cobjects
from annotation_pipeline.molecular_db_local import build_database_local
from annotation_pipeline.segment import define_ds_segments, chunk_spectra, segment_spectra, segment_centroids, \
    clip_centr_df, define_centr_segments, get_imzml_reader, validate_centroid_segments, validate_ds_segments
from annotation_pipeline.cache import PipelineCacher
from annotation_pipeline.segment_ds_vm import load_and_split_ds_vm
from annotation_pipeline.utils import PipelineStats, logger, upload_if_needed
from lithops.storage import Storage
from lithops.config import default_config


class Pipeline:
    def __init__(self, ds_config, db_config, use_db_cache=True, use_ds_cache=True, hybrid_impl='auto'):

        self.config = default_config()
        self.ds_config = ds_config
        self.db_config = db_config
        self.use_db_cache = use_db_cache
        self.use_ds_cache = use_ds_cache
        if hybrid_impl == 'auto':
            self.hybrid_impl = (
                self.config['lithops']['mode'] == 'localhost'
                or self.config['lithops']['mode'] == 'serverless' and 'ibm_vpc' in self.config
            )
            if self.hybrid_impl:
                logger.info(f'Using the Hybrid implementation')
            else:
                logger.info(f'Using the pure Serverless implementation')
        else:
            self.hybrid_impl = hybrid_impl

        lithops_bucket = self.config['lithops']['storage_bucket']
        self.ds_bucket = self.config.get('storage', {}).get('ds_bucket', lithops_bucket)

        self.lithops_executor = lithops.FunctionExecutor(config=self.config, runtime_memory=2048)
        if self.hybrid_impl:
            if self.config['lithops']['mode'] == 'localhost':
                self.lithops_vm_executor = self.lithops_executor
            else:
                self.lithops_vm_executor = lithops.StandaloneExecutor(config=self.config)

        self.storage = Storage(config=self.config)

        cache_namespace = 'vm' if hybrid_impl else 'function'
        self.cacher = PipelineCacher(
            self.storage, lithops_bucket, cache_namespace, self.ds_config["name"], self.db_config["name"]
        )
        if not self.use_db_cache or not self.use_ds_cache:
            self.cacher.clean(database=not self.use_db_cache, dataset=not self.use_ds_cache)

        stats_path_cache_key = ':ds/:db/stats_path.cache'
        if self.cacher.exists(stats_path_cache_key):
            self.stats_path = self.cacher.load(stats_path_cache_key)
            PipelineStats.path = self.stats_path
            logger.info(f'Using cached {self.stats_path} for statistics')
        else:
            PipelineStats.init()
            self.stats_path = PipelineStats.path
            self.cacher.save(self.stats_path, stats_path_cache_key)
            logger.info(f'Initialised {self.stats_path} for statistics')

        self.ds_segm_size_mb = 128
        self.image_gen_config = {
            "q": 99,
            "do_preprocessing": False,
            "nlevels": 30,
            "ppm": 3.0
        }

    def __call__(self, task='all', debug_validate=False):

        if task == 'all' or task == 'db':
            self.upload_molecular_databases()
            self.build_database(debug_validate=debug_validate)
            self.calculate_centroids(debug_validate=debug_validate)

        if task == 'all' or task == 'ds':
            self.upload_dataset()
            self.load_ds()
            self.split_ds()
            self.segment_ds(debug_validate=debug_validate)
            self.segment_centroids(debug_validate=debug_validate)
            self.annotate()
            self.run_fdr()

            if debug_validate and self.ds_config['metaspace_id']:
                self.check_results()

    def upload_dataset(self):
        self.imzml_cobject = upload_if_needed(
            self.storage, self.ds_config['imzml_path'], self.ds_bucket, 'imzml'
        )
        self.ibd_cobject = upload_if_needed(
            self.storage, self.ds_config['ibd_path'], self.ds_bucket, 'imzml'
        )

    def upload_molecular_databases(self, use_cache=True):
        cache_key = ':db/upload_molecular_databases.cache'

        if use_cache and self.cacher.exists(cache_key):
            self.mols_dbs_cobjects = self.cacher.load(cache_key)
            logger.info(f'Loaded {len(self.mols_dbs_cobjects)} molecular databases from cache')
        else:
            self.mols_dbs_cobjects = upload_mol_dbs_from_dir(self.storage, self.db_config['databases'])
            logger.info(f'Uploaded {len(self.mols_dbs_cobjects)} molecular databases')
            self.cacher.save(self.mols_dbs_cobjects, cache_key)

    def build_database(self, use_cache=True, debug_validate=False):
        if self.hybrid_impl:
            cache_key = ':ds/:db/build_database.cache'
            if use_cache and self.cacher.exists(cache_key):
                self.formula_cobjects, self.db_data_cobjects = self.cacher.load(cache_key)
                logger.info(f'Loaded {len(self.formula_cobjects)} formula segments and'
                            f' {len(self.db_data_cobjects)} db_data objects from cache')
            else:
                futures = self.lithops_vm_executor.call_async(
                    build_database_local,
                    (self.db_config, self.ds_config, self.mols_dbs_cobjects)
                )
                self.formula_cobjects, self.db_data_cobjects, build_db_exec_time = self.lithops_vm_executor.get_result(futures)
                PipelineStats.append_vm('build_database', build_db_exec_time,
                                        cloud_objects_n=len(self.formula_cobjects))
                logger.info(f'Built {len(self.formula_cobjects)} formula segments and'
                            f' {len(self.db_data_cobjects)} db_data objects')
                self.cacher.save((self.formula_cobjects, self.db_data_cobjects), cache_key)
        else:
            cache_key = ':db/build_database.cache'
            if use_cache and self.cacher.exists(cache_key):
                self.formula_cobjects, self.formula_to_id_cobjects = self.cacher.load(cache_key)
                logger.info(f'Loaded {len(self.formula_cobjects)} formula segments and'
                            f' {len(self.formula_to_id_cobjects)} formula-to-id chunks from cache')
            else:
                self.formula_cobjects, self.formula_to_id_cobjects = build_database(
                    self.lithops_executor, self.db_config, self.mols_dbs_cobjects
                )
                logger.info(f'Built {len(self.formula_cobjects)} formula segments and'
                            f' {len(self.formula_to_id_cobjects)} formula-to-id chunks')
                self.cacher.save((self.formula_cobjects, self.formula_to_id_cobjects), cache_key)

        if debug_validate:
            validate_formula_cobjects(self.storage, self.formula_cobjects)

    def calculate_centroids(self, use_cache=True, debug_validate=False):
        cache_key = ':ds/:db/calculate_centroids.cache'

        if use_cache and self.cacher.exists(cache_key):
            self.peaks_cobjects = self.cacher.load(cache_key)
            logger.info(f'Loaded {len(self.peaks_cobjects)} centroid chunks from cache')
        else:
            self.peaks_cobjects = calculate_centroids(
                self.lithops_executor, self.formula_cobjects, self.ds_config
            )
            logger.info(f'Calculated {len(self.peaks_cobjects)} centroid chunks')
            self.cacher.save(self.peaks_cobjects, cache_key)

        if debug_validate:
            validate_peaks_cobjects(self.lithops_executor, self.peaks_cobjects)

    def load_ds(self, use_cache=True):
        cache_key = ':ds/load_ds.cache'

        if self.hybrid_impl:
            pass  # all work is done in segment_ds
        else:
            if use_cache and self.cacher.exists(cache_key):
                self.imzml_reader, self.imzml_reader_cobject = self.cacher.load(cache_key)
                logger.info(f'Loaded imzml from cache, {len(self.imzml_reader.coordinates)} spectra found')
            else:
                self.imzml_reader, self.imzml_reader_cobject = get_imzml_reader(self.lithops_executor, self.imzml_cobject)
                logger.info(f'Parsed imzml: {len(self.imzml_reader.coordinates)} spectra found')
                self.cacher.save((self.imzml_reader, self.imzml_reader_cobject), cache_key)

    def split_ds(self, use_cache=True):
        cache_key = ':ds/split_ds.cache'

        if self.hybrid_impl:
            pass  # all work is done in segment_ds
        else:
            if use_cache and self.cacher.exists(cache_key):
                self.ds_chunks_cobjects = self.cacher.load(cache_key)
                logger.info(f'Loaded {len(self.ds_chunks_cobjects)} dataset chunks from cache')
            else:
                self.ds_chunks_cobjects = chunk_spectra(self.lithops_executor, self.ibd_cobject,
                                                        self.imzml_reader_cobject, self.imzml_reader)
                logger.info(f'Uploaded {len(self.ds_chunks_cobjects)} dataset chunks')
                self.cacher.save(self.ds_chunks_cobjects, cache_key)

    def segment_ds(self, use_cache=True, debug_validate=False):
        cache_key = ':ds/segment_ds.cache'

        if self.hybrid_impl:
            if use_cache and self.cacher.exists(cache_key):
                result = self.cacher.load(cache_key)
                logger.info(f'Loaded {len(result[2])} dataset segments from cache')
            else:
                sort_memory = 2**32
                fs = self.lithops_vm_executor.call_async(
                    load_and_split_ds_vm,
                    (self.imzml_cobject, self.ibd_cobject, self.ds_segm_size_mb, sort_memory),
                )
                result = self.lithops_vm_executor.get_result(fs)

                logger.info(f'Segmented dataset chunks into {len(result[2])} segments')
                self.cacher.save(result, cache_key)
            self.imzml_reader, \
            self.ds_segments_bounds, \
            self.ds_segms_cobjects, \
            self.ds_segms_len, \
            ds_segm_stats = result
            for func_name, exec_time in ds_segm_stats:
                if func_name == 'upload_segments':
                    cobjs_n = len(self.ds_segms_cobjects)
                else:
                    cobjs_n = 0
                PipelineStats.append_vm(func_name, exec_time, cloud_objects_n=cobjs_n)
        else:
            if use_cache and self.cacher.exists(cache_key):
                self.ds_segments_bounds, self.ds_segms_cobjects, self.ds_segms_len = \
                    self.cacher.load(cache_key)
                logger.info(f'Loaded {len(self.ds_segms_cobjects)} dataset segments from cache')
            else:
                sample_sp_n = 1000
                self.ds_segments_bounds = define_ds_segments(
                    self.lithops_executor,
                    self.ibd_cobject,
                    self.imzml_reader_cobject,
                    self.ds_segm_size_mb,
                    sample_sp_n,
                )
                self.ds_segms_cobjects, self.ds_segms_len = segment_spectra(
                    self.lithops_executor,
                    self.ds_chunks_cobjects,
                    self.ds_segments_bounds,
                    self.ds_segm_size_mb,
                    self.imzml_reader.mzPrecision,
                )
                logger.info(f'Segmented dataset chunks into {len(self.ds_segms_cobjects)} segments')
                self.cacher.save((self.ds_segments_bounds, self.ds_segms_cobjects, self.ds_segms_len), cache_key)

        self.ds_segm_n = len(self.ds_segms_cobjects)
        self.is_intensive_dataset = self.ds_segm_n * self.ds_segm_size_mb > 5000

        if debug_validate:
            validate_ds_segments(
                self.lithops_executor, self.imzml_reader, self.ds_segments_bounds,
                self.ds_segms_cobjects, self.ds_segms_len, self.hybrid_impl,
            )

    def segment_centroids(self, use_cache=True, debug_validate=False):
        mz_min, mz_max = self.ds_segments_bounds[0, 0], self.ds_segments_bounds[-1, 1]
        cache_key = ':ds/:db/segment_centroids.cache'

        if use_cache and self.cacher.exists(cache_key):
            self.clip_centr_chunks_cobjects, self.db_segms_cobjects = self.cacher.load(cache_key)
            logger.info(f'Loaded {len(self.db_segms_cobjects)} centroids segments from cache')
        else:
            self.clip_centr_chunks_cobjects, centr_n = \
                clip_centr_df(self.lithops_executor, self.peaks_cobjects, mz_min, mz_max)
            centr_segm_lower_bounds = define_centr_segments(self.lithops_executor, self.clip_centr_chunks_cobjects,
                                                            centr_n, self.ds_segm_n, self.ds_segm_size_mb)

            max_ds_segms_size_per_db_segm_mb = 2560 if self.is_intensive_dataset else 1536
            self.db_segms_cobjects = segment_centroids(self.lithops_executor, self.clip_centr_chunks_cobjects,
                                                       centr_segm_lower_bounds, self.ds_segments_bounds,
                                                       self.ds_segm_size_mb, max_ds_segms_size_per_db_segm_mb,
                                                       self.image_gen_config['ppm'])
            logger.info(f'Segmented centroids chunks into {len(self.db_segms_cobjects)} segments')

            self.cacher.save((self.clip_centr_chunks_cobjects, self.db_segms_cobjects), cache_key)

        self.centr_segm_n = len(self.db_segms_cobjects)

        if debug_validate:
            validate_centroid_segments(
                self.lithops_executor, self.db_segms_cobjects, self.ds_segments_bounds,
                self.image_gen_config['ppm']
            )

    def annotate(self, use_cache=True):
        cache_key = ':ds/:db/annotate.cache'

        if use_cache and self.cacher.exists(cache_key):
            self.formula_metrics_df, self.images_cloud_objs = self.cacher.load(cache_key)
            logger.info(f'Loaded {self.formula_metrics_df.shape[0]} metrics from cache')
        else:
            logger.info('Annotating...')
            if self.hybrid_impl:
                memory_capacity_mb = 2048 if self.is_intensive_dataset else 1024
            else:
                memory_capacity_mb = 4096 if self.is_intensive_dataset else 2048
            process_centr_segment = create_process_segment(self.ds_segms_cobjects,
                                                           self.ds_segments_bounds, self.ds_segms_len, self.imzml_reader,
                                                           self.image_gen_config, memory_capacity_mb, self.ds_segm_size_mb,
                                                           self.hybrid_impl)

            futures = self.lithops_executor.map(
                process_centr_segment,
                [co for co in self.db_segms_cobjects],
                runtime_memory=memory_capacity_mb,
            )
            formula_metrics_list, images_cloud_objs = zip(*self.lithops_executor.get_result(futures))
            self.formula_metrics_df = pd.concat(formula_metrics_list)
            self.images_cloud_objs = list(chain(*images_cloud_objs))
            PipelineStats.append_func(futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(self.images_cloud_objs))
            logger.info(f'Metrics calculated: {self.formula_metrics_df.shape[0]}')
            self.cacher.save((self.formula_metrics_df, self.images_cloud_objs), cache_key)

    def run_fdr(self, use_cache=True):
        cache_key = ':ds/:db/run_fdr.cache'

        if use_cache and self.cacher.exists(cache_key):
            self.fdrs = self.cacher.load(cache_key)
            logger.info('Loaded fdrs from cache')
        else:
            if self.hybrid_impl:
                futures = self.lithops_vm_executor.call_async(
                    calculate_fdrs_vm,
                    (self.formula_metrics_df, self.db_data_cobjects),
                )
                self.fdrs, fdr_exec_time = self.lithops_vm_executor.get_result(futures)

                PipelineStats.append_vm('calculate_fdrs', fdr_exec_time)
            else:
                rankings_df = build_fdr_rankings(
                    self.lithops_executor, self.ds_config, self.db_config, self.mols_dbs_cobjects,
                    self.formula_to_id_cobjects, self.formula_metrics_df
                )
                self.fdrs = calculate_fdrs(self.lithops_executor, rankings_df)
            self.cacher.save(self.fdrs, cache_key)

        logger.info('Number of annotations at with FDR less than:')
        for fdr_step in [0.05, 0.1, 0.2, 0.5]:
            logger.info(f'{fdr_step*100:2.0f}%: {(self.fdrs.fdr < fdr_step).sum()}')

    def get_results(self):
        self.results_df = (
            self.fdrs
            .join(self.formula_metrics_df)
            [lambda df: ~df.adduct.isna()]
            .sort_values('fdr')
            .assign(full_mol=lambda df: df.mol + df.modifier + df.adduct)
        )

        return self.results_df

    def get_images(self, as_png=True, only_first_isotope=True):
        # Only download interesting images, to avoid running out of memory
        targets = set(self.get_results().index[self.results_df.fdr <= 0.5])
        images = get_target_images(
            self.lithops_executor,
            self.images_cloud_objs,
            self.imzml_reader,
            targets,
            as_png=as_png,
            only_first_isotope=only_first_isotope,
        )

        return images

    def save_results(self, out_dir='.'):
        out_dir = Path(out_dir)
        images_dir = out_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        results_df = self.get_results()
        results_df.to_csv(out_dir / 'results.csv')
        image_sets = self.get_images(True, True)
        __import__('__main__').image_sets = image_sets

        filenames = (results_df.full_mol + '.png').to_dict()
        n_saved_images = 0
        for formula_i, image_set in image_sets.items():
            if image_set[0] is not None and formula_i in filenames:
                (images_dir / filenames[formula_i]).open('wb').write(image_set[0])
                n_saved_images += 1

        logger.info(f'Saved results.csv and {n_saved_images} images to {out_dir.resolve()}')

    def check_results(self):
        results_df = self.get_results()
        metaspace_options = self.config.get('metaspace_options', {})
        reference_results = get_reference_results(metaspace_options, self.ds_config['metaspace_id'])

        checked_results = check_results(results_df, reference_results)

        log_bad_results(**checked_results)
        return checked_results

    def clean(self, hard=False):
        self.cacher.clean(hard=hard)
