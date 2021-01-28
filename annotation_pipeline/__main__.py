import argparse
import json
import logging
from annotation_pipeline.pipeline import Pipeline

logger = logging.getLogger(name='annotation_pipeline')


def annotate(args, config):
    input_ds = json.load(args.ds)
    input_db = json.load(args.db)

    kwargs = {
        'use_db_cache': not args.no_cache,
        'use_ds_cache': not args.no_cache,
    }
    if args.impl != 'auto':
        kwargs.hybrid_impl = args.impl == 'hybrid'

    pipeline = Pipeline(input_ds, input_db, **kwargs)
    pipeline()
    if not args.no_output:
        pipeline.save_results(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run colocalization jobs',
                                     usage='python3 -m annotation_pipeline annotate [ds_config.json] [db_config.json] [output path]')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')

    subparsers = parser.add_subparsers(title='Sub-commands', dest='action')
    subparsers.required = True

    annotate_parser = subparsers.add_parser('annotate')
    annotate_parser.set_defaults(func=annotate)
    annotate_parser.add_argument('ds', type=argparse.FileType('r'), default='metabolomics/ds_config2.json',
                                 nargs='?', help='ds_config.json path')
    annotate_parser.add_argument('db', type=argparse.FileType('r'), default='metabolomics/db_config2.json',
                                 nargs='?', help='db_config.json path')
    annotate_parser.add_argument('output', default='output', nargs='?', help='directory to write output files')
    annotate_parser.add_argument('--no-output', action="store_true", help='prevents outputs from being written to file')
    annotate_parser.add_argument('--no-cache', action="store_true", help='prevents loading cached data from previous runs')
    annotate_parser.add_argument(
        '--impl',
        choices=['serverless', 'hybrid', 'auto'],
        default='auto',
        help='Selects whether to use the Serverless or Hybrid implementation. "auto" will select '
             'the Hybrid implementation if the selected platform is supported and correctly configured '
             '(running in localhost mode, or in serverless mode with ibm_vpc configured)'
    )

    args = parser.parse_args()
    config = json.load(args.config)
    args.func(args, config)

