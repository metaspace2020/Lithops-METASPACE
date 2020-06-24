from itertools import repeat
from pathlib import Path
from ibm_botocore.client import ClientError
from concurrent.futures import ThreadPoolExecutor
import msgpack_numpy as msgpack
import pandas as pd
import pickle
import hashlib
import math

from annotation_pipeline.formula_parser import safe_generate_ion_formula
from annotation_pipeline.utils import logger, get_ibm_cos_client, append_pywren_stats, \
    read_object_with_retry, read_cloud_object_with_retry

DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']
N_FORMULAS_SEGMENTS = 256
N_HASH_CHUNKS = 32  # should be less than N_FORMULAS_SEGMENTS


def build_database(pw, config, input_db):
    bucket = config["storage"]["db_bucket"]

    adducts = [*input_db['adducts'], *DECOY_ADDUCTS]
    modifiers = input_db['modifiers']
    databases = input_db['databases']

    def hash_formula_to_chunk(formula):
        m = hashlib.md5()
        m.update(formula.encode('utf-8'))
        return int(m.hexdigest(), 16) % N_HASH_CHUNKS

    def generate_formulas(adduct, storage):
        print(f'Generating formulas for adduct {adduct}')

        def _get_mols(mols_key):
            return pickle.loads(read_object_with_retry(storage, bucket, mols_key))

        with ThreadPoolExecutor(max_workers=128) as pool:
            mols_list = list(pool.map(_get_mols, databases))

        formulas = set()

        for mols in mols_list:
            for modifier in modifiers:
                formulas.update(map(safe_generate_ion_formula, mols, repeat(modifier), repeat(adduct)))

        if None in formulas:
            formulas.remove(None)

        formulas_chunks = {}
        for formula in formulas:
            chunk_i = hash_formula_to_chunk(formula)
            if chunk_i in formulas_chunks:
                formulas_chunks[chunk_i].append(formula)
            else:
                formulas_chunks[chunk_i] = [formula]

        def _store(chunk_i):
            return chunk_i, storage.put_cobject(pickle.dumps(formulas_chunks[chunk_i]))

        with ThreadPoolExecutor(max_workers=128) as pool:
            cobjects = dict(pool.map(_store, formulas_chunks.keys()))

        return cobjects

    memory_capacity_mb = 512
    futures = pw.map(generate_formulas, adducts, runtime_memory=memory_capacity_mb)
    results = pw.get_result(futures)
    chunk_cobjects = [[] for i in range(N_HASH_CHUNKS)]
    for cobjects_dict in results:
        for chunk_i, cobject in cobjects_dict:
            chunk_cobjects[chunk_i].append(cobject)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb,
                        cloud_objects_n=sum(map(len, chunk_cobjects)))

    def deduplicate_formulas_chunk(chunk_i, chunk_cobjects, storage):
        print(f'Deduplicating formulas chunk {chunk_i}')
        chunk = set()
        for cobject in chunk_cobjects:
            formulas_chunk_part = pickle.loads(read_cloud_object_with_retry(storage, cobject))
            chunk.update(formulas_chunk_part)

        return chunk

    def get_formulas_number_per_chunk(chunk_i, chunk_cobjects, storage):
        chunk = deduplicate_formulas_chunk(chunk_i, chunk_cobjects, storage)
        return len(chunk)

    memory_capacity_mb = 512
    futures = pw.map(get_formulas_number_per_chunk, enumerate(chunk_cobjects),
                     runtime_memory=memory_capacity_mb)
    formulas_nums = pw.get_result(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb)

    def store_formulas_segments(chunk_i, chunk_cobjects, storage):
        chunk = deduplicate_formulas_chunk(chunk_i, chunk_cobjects, storage)
        formula_i_start = sum(formulas_nums[:chunk_i])
        formula_i_end = formula_i_start + len(chunk)
        chunk = pd.DataFrame(sorted(chunk),
                            columns=['formula'],
                            index=pd.RangeIndex(formula_i_start, formula_i_end, name='formula_i'))

        n_threads = N_FORMULAS_SEGMENTS // N_HASH_CHUNKS
        segm_size = math.ceil(len(chunk) / n_threads)
        segm_list = [chunk[i:i+segm_size] for i in range(0, chunk.shape[0], segm_size)]

        def _store(segm_i):
            id = chunk_i * n_threads + segm_i
            print(f'Storing formulas segment {id}')
            return storage.put_object(segm_list[segm_i].to_msgpack())

        with ThreadPoolExecutor(max_workers=128) as pool:
            segm_cobjects = pool.map(_store, range(n_threads))

        return segm_cobjects

    memory_capacity_mb = 512
    futures = pw.map(store_formulas_segments, enumerate(chunk_cobjects),
                     runtime_memory=memory_capacity_mb)
    results = pw.get_result(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb,
                        cloud_objects_n=N_HASH_CHUNKS + N_FORMULAS_SEGMENTS)
    db_segm_cobjects = [segm for fdr, segms in results for segm in segms]

    num_formulas = sum(formulas_nums)
    n_formulas_chunks = sum([len(result) for result in results])
    logger.info(f'Generated {num_formulas} formulas in {n_formulas_chunks} chunks')

    formulas_bytes = 200 * num_formulas
    formula_to_id_chunk_mb = 512
    n_formula_to_id = int(math.ceil(formulas_bytes / (formula_to_id_chunk_mb * 1024 ** 2)))
    formula_to_id_bounds = [N_FORMULAS_SEGMENTS * ch_i // n_formula_to_id for ch_i in range(n_formula_to_id + 1)]
    formula_to_id_ranges = list(zip(formula_to_id_bounds[:-1], formula_to_id_bounds[1:]))
    formula_to_id_inputs = [db_segm_cobjects[start:end] for start, end in formula_to_id_ranges if start != end]

    def store_formula_to_id_chunk(ch_i, input_cobjects, storage):
        print(f'Storing formula_to_id dictionary chunk {ch_i}')

        def _get(cobj):
            formula_chunk = read_cloud_object_with_retry(storage, cobj, pd.read_msgpack)
            formula_to_id_chunk = dict(zip(formula_chunk.formula, formula_chunk.index))
            return formula_to_id_chunk

        formula_to_id = {}
        with ThreadPoolExecutor(max_workers=128) as pool:
            for chunk_dict in pool.map(_get, input_cobjects):
                formula_to_id.update(chunk_dict)

        return storage.put_cobject(msgpack.dumps(formula_to_id))

    safe_mb = 512
    memory_capacity_mb = formula_to_id_chunk_mb * 2 + safe_mb
    futures = pw.map(store_formula_to_id_chunk, enumerate(formula_to_id_inputs),
                     runtime_memory=memory_capacity_mb)
    formula_to_id_cobjects = pw.get_result(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, cloud_objects_n=n_formula_to_id)
    logger.info(f'Built {n_formula_to_id} formula_to_id dictionaries chunks')

    return db_segm_cobjects, formula_to_id_cobjects


def calculate_centroids(pw, db_segm_cobjects, ds_config):
    polarity = ds_config['polarity']
    isocalc_sigma = ds_config['isocalc_sigma']

    def calculate_peaks_for_formula(formula_i, formula):
        mzs, ints = isocalc_wrapper.centroids(formula)
        if mzs is not None:
            return list(zip(repeat(formula_i), range(len(mzs)), mzs, ints))
        else:
            return []

    def calculate_peaks_chunk(segm_i, segm_cobject, storage):
        print(f'Calculating peaks from formulas chunk {segm_i}')
        chunk_df = pd.read_msgpack(storage.get_cobject(segm_cobject, stream=True))
        peaks = [peak for formula_i, formula in chunk_df.formula.items()
                 for peak in calculate_peaks_for_formula(formula_i, formula)]
        peaks_df = pd.DataFrame(peaks, columns=['formula_i', 'peak_i', 'mz', 'int'])
        peaks_df.set_index('formula_i', inplace=True)

        print(f'Storing centroids chunk {id}')
        peaks_cobject = storage.put_cobject(peaks_df.to_msgpack())

        return peaks_cobject, peaks_df.shape[0]

    from annotation_pipeline.isocalc_wrapper import IsocalcWrapper # Import lazily so that the rest of the pipeline still works if the dependency is missing
    isocalc_wrapper = IsocalcWrapper({
        # These instrument settings are usually customized on a per-dataset basis out of a set of
        # 18 possible combinations, but most of EMBL's datasets are compatible with the following settings:
        'charge': {
            'polarity': polarity,
            'n_charges': 1,
        },
        'isocalc_sigma': float(f"{isocalc_sigma:f}") # Rounding to match production implementation
    })

    memory_capacity_mb = 2048
    futures = pw.map(calculate_peaks_chunk, enumerate(db_segm_cobjects), runtime_memory=memory_capacity_mb)
    results = pw.get_result(futures)
    append_pywren_stats(futures, memory_mb=memory_capacity_mb, cloud_objects_n=len(futures))

    num_centroids = sum(count for cobj, count in results)
    n_centroids_chunks = len(results)
    peaks_cobjects = [cobj for cobj, count in results]
    logger.info(f'Calculated {num_centroids} centroids in {n_centroids_chunks} chunks')
    return peaks_cobjects


def upload_mol_db_from_file(config, bucket, key, path, force=False):
    ibm_cos = get_ibm_cos_client(config)
    try:
        ibm_cos.head_object(Bucket=bucket, Key=key)
        should_dump = force
    except ClientError:
        should_dump = True

    if should_dump:
        mol_sfs = sorted(set(pd.read_csv(path).sf))
        ibm_cos.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(mol_sfs))


def upload_mol_dbs_from_dir(config, bucket, key_prefix, path, force=False):
    for file in Path(path).glob('mol_db*.csv'):
        key = f'{key_prefix}/{file.stem}.pickle'
        upload_mol_db_from_file(config, bucket, key, file, force)
