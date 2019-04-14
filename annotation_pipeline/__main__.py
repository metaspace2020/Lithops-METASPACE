import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from annotation_pipeline.pipeline import annotate_dataset


def annotate(args):
    config = json.load(args.config)
    input_config = json.load(args.input)
    input_data = input_config['dataset']
    input_db = input_config['molecular_db']
    output = Path(args.output) if not args.no_output else None

    if output and not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    formula_scores_df, formula_images = annotate_dataset(config, input_data, input_db)

    if output:
        formula_scores_df.to_pickle(output / 'formula_scores_df.pickle')
        for key, image_set in formula_images.items():
            for i, image in image_set:
                plt.imsave(output / f'{key}_{i}.png', image.toarray())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run colocalization jobs',
                                     usage='python -m annotation_pipeline annotate [input_config.json] [output path]')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')

    subparsers = parser.add_subparsers(title='Sub-commands', dest='action')
    subparsers.required = True

    annotate_parser = subparsers.add_parser('annotate')
    annotate_parser.add_argument('input', type=argparse.FileType('r'), default='input_config.json', nargs='?',
                                 help='input_config.json path')
    annotate_parser.add_argument('output', default='output', nargs='?', help='directory to write output files')
    annotate_parser.add_argument('--no-output', action="store_true", help='prevents outputs from being written to file')
    annotate_parser.set_defaults(func=annotate)
    subparsers.add_parser('process')
    subparsers.add_parser('convert')

    args = parser.parse_args()
    args.func(args)

