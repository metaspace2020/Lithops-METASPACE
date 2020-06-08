import argparse
import json
import os

from annotation_pipeline.utils import get_ibm_cos_client, upload_to_cos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload input data to COS', usage='')
    parser.add_argument('paths', type=str, nargs='+', help='path to upload [`tmp` is ignored]')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')
    args = parser.parse_args()

    config = json.load(args.config)
    cos_client = get_ibm_cos_client(config)

    for path in args.paths:
        for root, dirnames, filenames in os.walk(path):
            for fn in filenames:
                f_path = f'{root}/{fn}'
                upload_to_cos(cos_client, f_path, config['storage']['ds_bucket'], f_path)
