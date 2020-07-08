import argparse
import json
import time

from annotation_pipeline.pipeline import Pipeline
from annotation_pipeline.utils import get_pywren_stats

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run annotation pipeline', usage='')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')
    parser.add_argument('--ds', type=argparse.FileType('r'), default='metabolomics/ds_config2.json',
                        help='ds_config.json path')
    parser.add_argument('--db', type=argparse.FileType('r'), default='metabolomics/db_config2.json',
                        help='db_config.json path')

    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                        help='disable loading cached cloud objects')
    parser.set_defaults(use_cache=True)
    parser.add_argument('--no-vm', dest='use_vm', action='store_false',
                        help='run all steps with a serverless platform')
    parser.set_defaults(use_vm=True)
    parser.add_argument('--validate', dest='validate', action='store_true',
                        help='run validations on the pipeline\' steps')
    parser.set_defaults(validate=False)

    args = parser.parse_args()

    config = json.load(args.config)
    ds_config = json.load(args.ds)
    db_config = json.load(args.db)

    pipeline = Pipeline(config, ds_config, db_config, use_cache=args.use_cache, vm_algorithm=args.use_vm)

    start = time.time()
    pipeline(task='db', debug_validate=args.validate)
    print(f'--- database process: {time.time() - start:.2f} seconds ---')

    start = time.time()
    pipeline(task='ds', debug_validate=args.validate)
    print(f'--- dataset process: {time.time() - start:.2f} seconds ---')

    results_df = pipeline.get_results()
    stats = get_pywren_stats()
    print(stats)
