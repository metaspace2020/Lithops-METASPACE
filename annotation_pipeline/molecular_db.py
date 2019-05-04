import os
from io import BytesIO
from itertools import repeat

from ibm_botocore.client import ClientError
import pywren_ibm_cloud as pywren
import pandas as pd
import pickle

from .formula_parser import safe_generate_ion_formula
from .utils import get_ibm_cos_client


DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']


def calculate_centroids(config, input_db, formula_chunk_keys):
    def calculate_peaks_for_formula(formula_i, formula):
        mzs, ints = isocalc_wrapper.centroids(formula)
        if mzs is not None:
            return list(zip(repeat(formula_i), range(len(mzs)), mzs, ints))
        else:
            return []

    def calculate_peaks_for_chunk(key, data_stream):
        chunk_df = pd.read_pickle(data_stream._raw_stream, None)
        peaks = [peak for formula_i, formula in chunk_df.formula.items()
                 for peak in calculate_peaks_for_formula(formula_i, formula)]
        peaks_df = pd.DataFrame(peaks, columns=['formula_i', 'peak_i', 'mz', 'int'])
        return peaks_df

    def calculate_peaks_for_chunk_local(chunk_key, i):
        print(i, 'out of', len(formula_chunk_keys))
        chunk_df = pickle.loads(ibm_cos.get_object(Bucket=input_db["bucket"], Key=chunk_key)['Body'].read())
        peaks = [peak for formula_i, formula in chunk_df.formula.items()
                      for peak in calculate_peaks_for_formula(formula_i, formula)]
        peaks_df = pd.DataFrame(peaks, columns=['formula_i', 'peak_i', 'mz', 'int'])
        return peaks_df

    def merge_chunks_and_store(results, ibm_cos):
        centroids_df = pd.concat(results).set_index('formula_i')
        ibm_cos.put_object(Bucket=input_db['bucket'], Key=input_db['centroids_pandas'], Body=pickle.dumps(centroids_df))
        return centroids_df.shape, centroids_df.head(8)

    from .isocalc_wrapper import IsocalcWrapper # Import lazily so that the rest of the pipeline still works if the dependency is missing
    isocalc_wrapper = IsocalcWrapper({
        # These instrument settings are usually customized on a per-dataset basis out of a set of
        # 18 possible combinations, but most of EMBL's datasets are compatible with the following settings:
        'charge': {
            'polarity': '+',
            'n_charges': 1,
        },
        'isocalc_sigma': 0.001238
    })

    if True:
        # Change to false to calculate locally
        pw = pywren.ibm_cf_executor(config=config, runtime_memory=2048)
        iterdata = [f'{input_db["bucket"]}/{chunk_key}' for chunk_key in formula_chunk_keys]
        futures = pw.map_reduce(calculate_peaks_for_chunk, iterdata, merge_chunks_and_store)
        centroids_shape, centroids_head = pw.get_result(futures)
        pw.clean()
    else:
        ibm_cos = get_ibm_cos_client(config)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as ex:
            all_chunks = list(ex.map(calculate_peaks_for_chunk_local, formula_chunk_keys, range(len(formula_chunk_keys))))
        centroids_shape, centroids_head = merge_chunks_and_store(all_chunks, ibm_cos)

    return centroids_shape, centroids_head


def build_database(config, input_db, n_formula_chunks=256):
    def generate_formulas(key, data_stream, adducts, modifiers):
        mols = pickle.loads(data_stream.read())
        formulas = set()

        for adduct in adducts:
            for modifier in modifiers:
                formulas.update(map(safe_generate_ion_formula, mols, repeat(modifier), repeat(adduct)))

        if None in formulas:
            formulas.remove(None)
        return formulas

    def store_formulas(results, ibm_cos):
        # Reduce list of formulas for processing to include only unique formulas
        formulas = pd.DataFrame(sorted(set().union(*results)), columns=['formula'])
        formulas.index.name = 'formula_i'

        num_chunks = min(len(formulas), n_formula_chunks)
        chunk_keys = []

        # Write pickled chunks equivalent to formulas.csv with (formula_i, formula)
        for i in range(num_chunks):
            lo = len(formulas) * i // num_chunks
            hi = len(formulas) * (i + 1) // num_chunks
            chunk_key = f'{input_db["formulas_chunks"]}/{i}.pickle'
            chunk_keys.append(chunk_key)
            chunk = pd.DataFrame(formulas.formula[lo:hi], copy=True)
            chunk.index.name = 'formula_i'

            ibm_cos.put_object(Bucket=input_db['bucket'], Key=chunk_key,
                               Body=pickle.dumps(chunk))

        return len(formulas), chunk_keys

    adducts = [*input_db['adducts'], *DECOY_ADDUCTS]
    modifiers = input_db['modifiers']
    databases = input_db['databases']

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=2048)
    iterdata = [(f'{input_db["bucket"]}/{database}', [adduct], modifiers) for database in databases for adduct in adducts]
    futures = pw.map_reduce(generate_formulas, iterdata, store_formulas)
    num_formulas, formula_chunk_keys = pw.get_result(futures)
    pw.clean()

    return num_formulas, formula_chunk_keys


def clean_formula_chunks(config, input_db, formula_chunk_keys):
    ibm_cos = get_ibm_cos_client(config)
    ibm_cos.delete_objects(Bucket=input_db['bucket'],
                           Delete={'Objects': [{'Key': key} for key in formula_chunk_keys]})


def dump_mol_db(config, bucket, key, db_id, force=False):
    import requests
    ibm_cos = get_ibm_cos_client(config)
    try:
        ibm_cos.head_object(Bucket=bucket, Key=key)
        should_dump = force
    except ClientError:
        should_dump = True

    if should_dump:
        mols = requests.get(f'https://metaspace2020.eu/mol_db/v1/databases/{db_id}/molecules?limit=999999&fields=sf').json()['data']
        mols_df = sorted(set(mol['sf'] for mol in mols))
        with BytesIO() as fileobj:
            ibm_cos.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(mols_df))