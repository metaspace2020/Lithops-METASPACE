from itertools import product, repeat
import pickle
import numpy as np
import pandas as pd
import msgpack_numpy as msgpack

from annotation_pipeline.formula_parser import safe_generate_ion_formula
from annotation_pipeline.molecular_db import DECOY_ADDUCTS
from annotation_pipeline.utils import append_pywren_stats, read_object_with_retry, list_keys


def _get_random_adduct_set(size, adducts, offset):
    r = np.random.RandomState(123)
    idxs = (r.random_integers(0, len(adducts), size) + offset) % len(adducts)
    return np.array(adducts)[idxs]


def msgpack_load_text(stream):
    return msgpack.load(stream, encoding='utf-8')


def build_fdr_rankings(pw, bucket, input_data, input_db, formula_scores_df):

    def build_ranking(group_i, ranking_i, database, modifier, adduct, id, ibm_cos):
        print("Building ranking...")
        print(f'job_i: {id}')
        print(f'ranking_i: {ranking_i}')
        print(f'database: {database}')
        print(f'modifier: {modifier}')
        print(f'adduct: {adduct}')
        # For every unmodified formula in `database`, look up the MSM score for the molecule
        # that it would become after the modifier and adduct are applied
        mols = pickle.loads(read_object_with_retry(ibm_cos, bucket, database))
        if adduct is not None:
            # Target rankings use the same adduct for all molecules
            mol_formulas = list(map(safe_generate_ion_formula, mols, repeat(modifier), repeat(adduct)))
        else:
            # Decoy rankings use a consistent random adduct for each molecule, chosen so that it doesn't overlap
            # with other decoy rankings for this molecule
            adducts = _get_random_adduct_set(len(mols), decoy_adducts, ranking_i)
            mol_formulas = list(map(safe_generate_ion_formula, mols, repeat(modifier), adducts))

        formula_to_id = {}
        keys = list_keys(bucket, f'{input_db["formula_to_id_chunks"]}/', ibm_cos)
        for key in keys:
            formula_to_id_chunk = read_object_with_retry(ibm_cos, bucket, key, msgpack_load_text)

            for formula in mol_formulas:
                if formula_to_id_chunk.get(formula) is not None:
                    formula_to_id[formula] = formula_to_id_chunk.get(formula)

        formula_is = [formula and formula_to_id.get(formula) for formula in mol_formulas]
        msm = [formula_i and msm_lookup.get(formula_i) for formula_i in formula_is]
        if adduct is not None:
            ranking_df = pd.DataFrame({'mol': mols, 'msm': msm}, index=formula_is)
            ranking_df = ranking_df[~ranking_df.msm.isna()]
            key = f'{input_data["fdr_rankings"]}/{group_i}/target{ranking_i}.pickle'
        else:
            # Specific molecules don't matter in the decoy rankings, only their msm distribution
            ranking_df = pd.DataFrame({'msm': msm})
            ranking_df = ranking_df[~ranking_df.msm.isna()]
            key = f'{input_data["fdr_rankings"]}/{group_i}/decoy{ranking_i}.pickle'

        ibm_cos.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(ranking_df))
        return id, key

    decoy_adducts = sorted(set(DECOY_ADDUCTS).difference(input_db['adducts']))
    n_decoy_rankings = input_data.get('num_decoys', len(decoy_adducts))
    msm_lookup = formula_scores_df.msm.to_dict() # Ideally this data would stay in COS so it doesn't have to be reuploaded

    # Create a job for each list of molecules to be ranked
    ranking_jobs = []
    for group_i, (database, modifier) in enumerate(product(input_db['databases'], input_db['modifiers'])):
        # Target and decoy rankings are treated differently. Decoy rankings are identified by not having an adduct.
        ranking_jobs.extend((group_i, ranking_i, database, modifier, adduct)
                             for ranking_i, adduct in enumerate(input_db['adducts']))
        ranking_jobs.extend((group_i, ranking_i, database, modifier, None)
                             for ranking_i in range(n_decoy_rankings))

    memory_capacity_mb = 1024
    futures = pw.map(build_ranking, ranking_jobs, runtime_memory=memory_capacity_mb)
    ranking_keys = [key for job_i, key in sorted(pw.get_result(futures))]
    append_pywren_stats(futures, memory=memory_capacity_mb, plus_objects=len(futures))

    rankings_df = pd.DataFrame(ranking_jobs, columns=['group_i', 'ranking_i', 'database_path', 'modifier', 'adduct'])
    rankings_df = rankings_df.assign(is_target=~rankings_df.adduct.isnull(), key=ranking_keys)

    return rankings_df


def calculate_fdrs(pw, data_bucket, rankings_df):

    def run_ranking(ibm_cos, data_bucket, target_key, decoy_key):
        target = pickle.loads(read_object_with_retry(ibm_cos, data_bucket, target_key))
        decoy = pickle.loads(read_object_with_retry(ibm_cos, data_bucket, decoy_key))
        merged = pd.concat([target.assign(is_target=1), decoy.assign(is_target=0)], sort=False)
        merged = merged.sort_values('msm', ascending=False)
        decoy_cumsum = (merged.is_target == False).cumsum()
        target_cumsum = merged.is_target.cumsum()
        base_fdr = np.clip(decoy_cumsum / target_cumsum, 0, 1)
        base_fdr[np.isnan(base_fdr)] = 1
        target_fdrs = merged.assign(fdr=base_fdr)[lambda df: df.is_target == 1]
        target_fdrs = target_fdrs.drop('is_target', axis=1)
        target_fdrs = target_fdrs.sort_values('msm')
        target_fdrs = target_fdrs.assign(fdr=np.minimum.accumulate(target_fdrs.fdr))
        target_fdrs = target_fdrs.sort_index()
        return target_fdrs

    def merge_rankings(ibm_cos, data_bucket, target_row, decoy_keys):
        print("Merging rankings...")
        print(target_row)
        rankings = [run_ranking(ibm_cos, data_bucket, target_row.key, decoy_key) for decoy_key in decoy_keys]
        mols = (pd.concat(rankings)
                .rename_axis('formula_i')
                .reset_index()
                .groupby('formula_i')
                .agg({'fdr': np.nanmedian, 'mol': 'first'})
                .assign(database_path=target_row.database_path,
                        adduct=target_row.adduct,
                        modifier=target_row.modifier))
        return mols

    ranking_jobs = []
    for group_i, group in rankings_df.groupby('group_i'):
        target_rows = group[group.is_target]
        decoy_rows = group[~group.is_target]

        for i, target_row in target_rows.iterrows():
            ranking_jobs.append([data_bucket, target_row, decoy_rows.key.tolist()])

    memory_capacity_mb = 256
    futures = pw.map(merge_rankings, ranking_jobs, runtime_memory=memory_capacity_mb)
    results = pw.get_result(futures)
    append_pywren_stats(futures, memory=memory_capacity_mb)

    return pd.concat(results)
