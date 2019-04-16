import os
from itertools import repeat

import ibm_boto3
from ibm_botocore.client import Config
import pywren_ibm_cloud as pywren
import pandas as pd
import pickle

from .formula_parser import safe_generate_ion_formula


DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']

def get_cos_client(config):
    return ibm_boto3.client(service_name='s3',
                            ibm_api_key_id=config['ibm_cos']['api_key'],
                            config=Config(signature_version='oauth'),
                            endpoint_url=config['ibm_cos']['endpoint'])


def process_formulas_database(config, input_db):
    def process_formulas(key, data_stream):
        formulas_df = pd.read_csv(data_stream._raw_stream).set_index('formula_i')
        return formulas_df.shape, formulas_df.head()

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=256)
    iterdata = [f'{input_db["bucket"]}/{input_db["formulas"]}']
    pw.map(process_formulas, iterdata)
    formulas_shape, formulas_head = pw.get_result()
    pw.clean()

    return formulas_shape, formulas_head


def store_centroids_database(config, input_db):
    def store_centroids(key, data_stream, ibm_cos):
        centroids_df = pd.read_csv(data_stream._raw_stream).set_index('formula_i')
        ibm_cos.put_object(Bucket=input_db['bucket'], Key=input_db['centroids_pandas'], Body=pickle.dumps(centroids_df))
        return centroids_df.shape, centroids_df.head(8)

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=1024)
    iterdata = [f'{input_db["bucket"]}/{input_db["centroids"]}']
    pw.map(store_centroids, iterdata)
    centroids_shape, centroids_head = pw.get_result()
    pw.clean()

    return centroids_shape, centroids_head


def calculate_centroids(config, input_db, formula_chunk_keys):
    def calculate_peaks_for_formula(formula_i, formula):
        mzs, ints = isocalc_wrapper.centroids(formula)
        if mzs is not None:
            return list(zip(repeat(formula_i), range(len(mzs)), mzs, ints))
        else:
            return []

    def calculate_peaks_for_chunk(key, data_stream):
        chunk_df = pd.read_pickle(data_stream._raw_stream)
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

    def merge_chunks_and_store(chunks, ibm_cos):
        centroids_df = pd.concat(chunks).set_index('formula_i')
        ibm_cos.put_object(Bucket=input_db['bucket'], Key=input_db['centroids_pandas'], Body=pickle.dumps(centroids_df))
        return centroids_df.shape, centroids_df.head(8)

    from .isocalc_wrapper import IsocalcWrapper # Import lazily so that the rest of the pipeline still works if the dependency is missing
    isocalc_wrapper = IsocalcWrapper({
        # These instrument settings are usually customized on a per-dataset basis,
        # but most of EMBL's datasets use these settings:
        'charge': {
            'polarity': '+',
            'n_charges': 1,
        },
        'isocalc_sigma': 0.001238
    })

    # TODO: Switch to pywren codepath when cpyMSpec is in the runtime
    # pw = pywren.ibm_cf_executor(config=config, runtime_memory=1024)
    # iterdata = [f'{input_db["bucket"]}/{chunk_key}' for chunk_key in formula_chunk_keys]
    # futures = pw.map_reduce(calculate_peaks_for_chunk, iterdata, merge_chunks_and_store)
    # centroids_shape, centroids_head = pw.get_result(futures)
    # pw.clean()

    ibm_cos = get_cos_client(config)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as ex:
        all_chunks = list(ex.map(calculate_peaks_for_chunk_local, formula_chunk_keys, range(len(formula_chunk_keys))))
    centroids_shape, centroids_head = merge_chunks_and_store(all_chunks, ibm_cos)

    return centroids_shape, centroids_head


def build_database(config, input_db, n_formula_chunks=256):
    def generate_formulas(key, data_stream, adducts, modifiers):
        db = pd.read_csv(data_stream._raw_stream)
        dfs = []

        for adduct in adducts:
            for modifier in modifiers:
                formulas = db.apply(lambda row: safe_generate_ion_formula(row.sf, adduct, modifier), axis=1)
                df = db.assign(formula=formulas, adduct=adduct, modifier=modifier)
                df = df[formulas != None]
                dfs.append(df)

        result = pd.concat(dfs)

        return result

    def store_formulas(results, ibm_cos):
        ions = pd.concat(results)
        ions.sort_values(['sf','adduct','modifier'], inplace=True)
        ions.reset_index(drop=True, inplace=True)
        ions.index.name = 'ion_i'

        # Reduce list of formulas for processing to include only unique formulas
        formulas = ions[['formula']].drop_duplicates().sort_values(by='formula').reset_index(drop=True)
        formulas.index.name = 'formula_i'

        # Add formula IDs to ions so that ions can be mapped back to formulas
        ions = ions.merge(formulas.reset_index(), on='formula', how='outer')

        num_chunks = min(len(formulas), n_formula_chunks)
        chunk_keys = []

        # Write pickled chunks equivalent to formulas.csv with (formula_i, formula)
        for i in range(num_chunks):
            lo = len(formulas) * i // num_chunks
            hi = len(formulas) * (i + 1) // num_chunks
            chunk_key = f'{input_db["formulas_dir"]}/{i}.pickle'
            chunk_keys.append(chunk_key)
            chunk = pd.DataFrame(formulas.formula[lo:hi], copy=True)
            chunk.index.name = 'formula_i'

            ibm_cos.put_object(Bucket=input_db['bucket'], Key=chunk_key,
                               Body=pickle.dumps(chunk))

        # Write full dataframe to pickle so that molecules can be matched after annotation for further analysis
        ibm_cos.put_object(Bucket=input_db['bucket'], Key=input_db['ions_full'],
                           Body=pickle.dumps(ions))

        return chunk_keys

    adducts = [*input_db.get('adducts', ['+H', '+Na', '+K']), *DECOY_ADDUCTS]
    modifiers = input_db.get('modifiers') or ['']
    databases = input_db['databases']

    ibm_cos = get_cos_client(config)
    pw = pywren.ibm_cf_executor(config=config, runtime_memory=512)
    iterdata = [(f'{input_db["bucket"]}/{database}', [adduct], modifiers) for database in databases for adduct in adducts]
    # TODO: Fix OOM in reducer. Maybe by handling ions_full separately because it's much bigger and not needed until after FDR ranking
    # futures = pw.map_reduce(generate_formulas, iterdata, store_formulas)
    # formula_chunk_keys = pw.get_result(futures)
    futures = pw.map(generate_formulas, iterdata)
    results = pw.get_result(futures)
    formula_chunk_keys = store_formulas(results, ibm_cos)
    pw.clean()

    return formula_chunk_keys


def clean_formula_chunks(config, input_db, formula_chunk_keys):
    ibm_cos = get_cos_client(config)
    ibm_cos.delete_objects(Bucket=input_db['bucket'],
                           Delete={'Objects': [{'Key': key} for key in formula_chunk_keys]})


def dump_mol_db_to_file(db_id, file):
    import requests
    mols = requests.get(f'https://metaspace2020.eu/mol_db/v1/databases/{db_id}/molecules?limit=999999&fields=sf').json()['data']
    mols_df = pd.DataFrame([mol['sf'] for mol in mols], columns=[['sf']]).drop_duplicates()
    mols_df.to_csv(file)