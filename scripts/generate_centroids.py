import argparse
import json
import pandas as pd
import time

from click import Path

from annotation_pipeline.__main__ import get_ibm_cos_client
from annotation_pipeline.molecular_db import dump_mol_db, build_database, calculate_centroids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate centroids', usage='')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')
    parser.add_argument('--input-config', type=argparse.FileType('r'), default='input_config.json',
                        help='input_config.json path')
    args = parser.parse_args()

    start = time.time()

    input_config = json.load(args.input_config)
    input_db = input_config['molecular_db']

    config = json.load(args.config)
    cos_client = get_ibm_cos_client(config)

    databases_path = Path(input_db['databases'][0].parent)
    dump_mol_db(config, config['storage']['db_bucket'], f'{databases_path}/mol_db1.pickle', 22)  # HMDB-v4
    dump_mol_db(config, config['storage']['db_bucket'], f'{databases_path}/mol_db2.pickle', 19)  # ChEBI-2018-01
    dump_mol_db(config, config['storage']['db_bucket'], f'{databases_path}/mol_db3.pickle', 24)  # LipidMaps-2017-12-12
    dump_mol_db(config, config['storage']['db_bucket'], f'{databases_path}/mol_db4.pickle', 26)  # SwissLipids-2018-02-02

    num_formulas, chunk_keys = build_database(config, input_db)
    print(f'Number of formulas: {num_formulas}')
    centroids_shape, centroids_head = calculate_centroids(config, input_db, chunk_keys)
    print(f'Number of centroids generated: {centroids_shape[0]}')

    resp = cos_client.get_object(Bucket=config['storage']['db_bucket'], Key=input_db['centroids_pandas'])
    with open(input_db['centroids_pandas'], 'wb') as f:
        f.write(resp['Body'].read())
    centroids_df = pd.read_pickle(input_db['centroids_pandas'])
    print(f'Number of centroids downloaded: {centroids_df.shape[0]}')

    print(f'--- {time.time() - start:.2f} seconds ---')
