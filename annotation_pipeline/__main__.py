import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from annotation_pipeline.imzml import convert_imzml_to_txt
from annotation_pipeline.pipeline import annotate_dataset

logger = logging.getLogger(name='annotation_pipeline')


def get_ibm_cos_client(config):
    import ibm_boto3
    from ibm_botocore.client import Config
    return ibm_boto3.client(service_name='s3',
                            ibm_api_key_id=config['ibm_cos']['api_key'],
                            config=Config(signature_version='oauth'),
                            endpoint_url=config['ibm_cos']['endpoint'])


def annotate(args, config):
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


def convert_imzml(args, config):
    assert args.input.endswith('.imzML')
    assert args.output.endswith('.txt')

    if args.cos_input or args.cos_output:
        temp_dir = TemporaryDirectory()
        ibm_cos = get_ibm_cos_client(config)

    # Download input if using COS
    if args.cos_input:
        logger.info('Downloading input files')
        input_bucket, input_key_imzml = args.input.split('/', 1)
        imzml_filename = input_key_imzml.split('/')[-1]
        imzml_path = str(Path(temp_dir.name) / imzml_filename)

        logger.debug('download_file', input_bucket, input_key_imzml, imzml_path)
        ibm_cos.download_file(input_bucket, input_key_imzml, imzml_path)

        input_key_ibd = input_key_imzml[:-6] + '.ibd'
        ibd_path = imzml_path[:-6] + '.ibd'
        logger.debug('download_file', input_bucket, input_key_ibd, ibd_path)
        ibm_cos.download_file(input_bucket, input_key_ibd, ibd_path)
    else:
        imzml_path = args.input

    # Generate local path for output if using COS
    if args.cos_output:
        output_bucket, output_key = args.output.split('/', 1)
        spectra_filename = output_key.split('/')[-1]
        spectra_path = str(Path(temp_dir.name) / spectra_filename)
    else:
        spectra_path = args.output
    coord_path = spectra_path[:-4] + '_coord.txt'

    logger.info('Converting to txt')
    logger.debug('convert_imzml_to_txt', imzml_path, spectra_path, coord_path)
    convert_imzml_to_txt(imzml_path, spectra_path, coord_path)

    # Upload output if using COS
    if args.cos_output:
        logger.info('Uploading output files')
        logger.debug('upload_file', output_bucket, output_key, spectra_path)
        ibm_cos.upload_file(spectra_path, output_bucket, output_key)

        output_key_coord = output_key[:-4] + '_coord.txt'
        logger.debug('upload_file', output_bucket, output_key_coord, coord_path)
        ibm_cos.upload_file(coord_path, output_bucket, output_key_coord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run colocalization jobs',
                                     usage='python -m annotation_pipeline annotate [input_config.json] [output path]')
    parser.add_argument('--config', type=argparse.FileType('r'), default='config.json', help='config.json path')

    subparsers = parser.add_subparsers(title='Sub-commands', dest='action')
    subparsers.required = True

    annotate_parser = subparsers.add_parser('annotate')
    annotate_parser.set_defaults(func=annotate)
    annotate_parser.add_argument('input', type=argparse.FileType('r'), default='input_config.json', nargs='?',
                                 help='input_config.json path')
    annotate_parser.add_argument('output', default='output', nargs='?', help='directory to write output files')
    annotate_parser.add_argument('--no-output', action="store_true", help='prevents outputs from being written to file')

    convert_parser = subparsers.add_parser('convert_imzml')
    convert_parser.set_defaults(func=convert_imzml)
    convert_parser.add_argument('input', help='path to .imzML file (matching .ibd file must be in the same directory)')
    convert_parser.add_argument('output', default='ds.txt', nargs='?',
                                help='output spectra txt file (matching coord file will be created in same directory)')
    convert_parser.add_argument('--cos-input', action='store_true',
                                help='Indicates the input files should be downloaded from IBM COS. '
                                     'The input path should be in "bucket/key" format.')
    convert_parser.add_argument('--cos-output', action='store_true',
                                help='Indicates the output files should be uploaded to IBM COS. '
                                     'The output path should be in "bucket/key" format.')

    args = parser.parse_args()
    config = json.load(args.config)
    args.func(args, config)
