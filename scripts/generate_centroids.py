import argparse
import json
import time

from annotation_pipeline.__main__ import generate_centroids
from annotation_pipeline.utils import get_pywren_stats

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate centroids', usage='')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')
    parser.add_argument('--ds', type=argparse.FileType('r'), default='metabolomics/ds_config2.json',
                        help='ds_config.json path')
    parser.add_argument('--db', type=argparse.FileType('r'), default='metabolomics/db_config2.json',
                        help='db_config.json path')
    args = parser.parse_args()
    config = json.load(args.config)

    start = time.time()
    generate_centroids(args, config)
    print(f'--- {time.time() - start:.2f} seconds ---')

    stats = get_pywren_stats()
    print(stats)
