import argparse
import json
import time

from annotation_pipeline_v2.pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run annotation pipeline', usage='')
    parser.add_argument('--input-config', type=argparse.FileType('r'), default='input_config.json',
                        help='input_config.json path')
    args = parser.parse_args()

    start = time.time()

    input_config = json.load(args.input_config)

    pipeline = Pipeline(input_config)
    pipeline.load_ds()
    pipeline.segment_ds()
    pipeline.segment_centroids()
    formula_metrics_df, formula_images = pipeline.annotate()

    print(f'--- {time.time() - start:.2f} seconds ---')
