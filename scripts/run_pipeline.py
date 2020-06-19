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
                        help='disable loading and saving cached cloud objects')
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    config = json.load(args.config)
    ds_config = json.load(args.ds)
    db_config = json.load(args.db)

    pipeline = Pipeline(config, ds_config, db_config, use_cache=args.use_cache)

    start = time.time()

    pipeline()
    results_df = pipeline.get_results()

    print(f'--- {time.time() - start:.2f} seconds ---')

    stats = get_pywren_stats()
    print(stats)
