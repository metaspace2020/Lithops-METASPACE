import argparse
import json
import time

from annotation_pipeline.__main__ import generate_centroids

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate centroids', usage='')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')
    parser.add_argument('--input', type=argparse.FileType('r'), default='input_config.json',
                        help='input_config.json path')
    args = parser.parse_args()

    start = time.time()

    config = json.load(args.config)
    generate_centroids(args, config)

    print(f'--- {time.time() - start:.2f} seconds ---')
