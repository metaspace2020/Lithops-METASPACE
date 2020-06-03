from itertools import repeat, product
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import msgpack_numpy as msgpack
import pandas as pd
import pickle
import math

from annotation_pipeline.formula_parser import safe_generate_ion_formula
from annotation_pipeline.utils import logger, clean_from_cos, read_object_with_retry


DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']
N_FORMULAS_SEGMENTS = 256
FORMULA_TO_ID_CHUNK_MB = 512


def build_database_local(storage, config, input_db):
    bucket = config["storage"]["db_bucket"]
    formulas_chunks_prefix = input_db["formulas_chunks"]
    formula_to_id_chunks_prefix = input_db["formula_to_id_chunks"]
    clean_from_cos(config, bucket, formulas_chunks_prefix)
    clean_from_cos(config, bucket, formula_to_id_chunks_prefix)

    formulas_df = get_formulas_df(storage, bucket, input_db)
    num_formulas = len(formulas_df)
    logger.info(f'Generated {num_formulas} formulas')

    n_formulas_chunks = store_formula_segments(storage, bucket, formulas_chunks_prefix, formulas_df)
    logger.info(f'Stored {num_formulas} formulas in {n_formulas_chunks} chunks')

    n_formula_to_id = store_formula_to_id(storage, bucket, formula_to_id_chunks_prefix, formulas_df)
    logger.info(f'Built {n_formula_to_id} formula_to_id dictionaries chunks')

    return num_formulas, n_formulas_chunks


def _generate_formulas(args):
    mols, modifier, adduct = args
    return list(map(safe_generate_ion_formula, mols, repeat(modifier), repeat(adduct)))


def get_formulas_df(storage, bucket, input_db):
    adducts = [*input_db['adducts'], *DECOY_ADDUCTS]
    modifiers = input_db['modifiers']
    databases = input_db['databases']

    # Load databases
    def _get_mols(mols_key):
        return pickle.loads(read_object_with_retry(storage, bucket, mols_key))

    with ThreadPoolExecutor(max_workers=128) as pool:
        dbs = list(pool.map(_get_mols, databases))

    # Calculate formulas
    formulas = set()
    with ProcessPoolExecutor() as ex:
        for chunk in ex.map(_generate_formulas, product(dbs, modifiers, adducts)):
            formulas.update(chunk)

    if None in formulas:
        formulas.remove(None)

    return pd.DataFrame({'formula': sorted(formulas)}).rename_axis(index='formula_i')


def store_formula_segments(storage, bucket, formulas_chunks_prefix, formulas_df):
    subsegm_size = math.ceil(len(formulas_df) / N_FORMULAS_SEGMENTS)
    segm_list = [formulas_df[i:i+subsegm_size] for i in range(0, len(formulas_df), subsegm_size)]

    def _store(segm_i):
        storage.put_object(Bucket=bucket,
                           Key=f'{formulas_chunks_prefix}/{segm_i}.msgpack',
                           Body=segm_list[segm_i].to_msgpack())

    with ThreadPoolExecutor(max_workers=128) as pool:
        pool.map(_store, range(len(segm_list)))

    return len(segm_list)


def store_formula_to_id(storage, bucket, formula_to_id_chunks_prefix, formulas_df):
    num_formulas = len(formulas_df)
    n_formula_to_id = int(math.ceil(num_formulas * 200 / (FORMULA_TO_ID_CHUNK_MB * 1024 ** 2)))
    for ch_i in range(n_formula_to_id):
        print(f'Storing formula_to_id dictionary chunk {ch_i}')
        start_idx = num_formulas * ch_i // n_formula_to_id
        end_idx = num_formulas * (ch_i + 1) // n_formula_to_id

        formula_to_id = formulas_df.iloc[start_idx:end_idx].formula.to_dict()

        storage.put_object(Bucket=bucket,
                           Key=f'{formula_to_id_chunks_prefix}/{ch_i}.msgpack',
                           Body=msgpack.dumps(formula_to_id))
    return n_formula_to_id
