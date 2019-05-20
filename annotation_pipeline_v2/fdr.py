from itertools import product, repeat
import pickle
import numpy as np
import pandas as pd
import pywren_ibm_cloud as pywren

from annotation_pipeline.formula_parser import safe_generate_ion_formula
from annotation_pipeline.molecular_db import DECOY_ADDUCTS


def _get_random_adduct_set(size, adducts, offset):
    r = np.random.RandomState(123)
    idxs = (r.random_integers(0, len(adducts), size) + offset) % len(adducts)
    return np.array(adducts)[idxs]


def build_fdr_rankings(config, input_data, input_db, formula_scores_df):

    def build_ranking(key, data_stream, ibm_cos, job_i, group_i, ranking_i, database, modifier, adduct):
        # For every unmodified formula in `database`, look up the MSM score for the molecule
        # that it would become after the modifier and adduct are applied
        formula_to_id = pickle.loads(data_stream.read())
        mols = pickle.loads(ibm_cos.get_object(Bucket=input_db["bucket"], Key=database)['Body'].read())
        if adduct is not None:
            # Target rankings use the same adduct for all molecules
            mol_formulas = list(map(safe_generate_ion_formula, mols, repeat(modifier), repeat(adduct)))
        else:
            # Decoy rankings use a consistent random adduct for each molecule, chosen so that it doesn't overlap
            # with other decoy rankings for this molecule
            adducts = _get_random_adduct_set(len(mols), decoy_adducts, ranking_i)
            mol_formulas = list(map(safe_generate_ion_formula, mols, repeat(modifier), adducts))

        formula_is = [formula and formula_to_id.get(formula) for formula in mol_formulas]
        msm = [formula_i and msm_lookup.get(formula_i) for formula_i in formula_is]
        if adduct is not None:
            ranking_df = pd.DataFrame({'mol': mols, 'msm': msm}, index=formula_is)
            key = f'{input_data["fdr_rankings"]}/{group_i}/target{ranking_i}.pickle'
        else:
            # Specific molecules don't matter in the decoy rankings, only their msm distribution
            ranking_df = pd.DataFrame({'msm': msm})
            key = f'{input_data["fdr_rankings"]}/{group_i}/decoy{ranking_i}.pickle'

        ibm_cos.put_object(Bucket=input_data["bucket"], Key=key, Body=pickle.dumps(ranking_df))
        return job_i, key

    formula_to_id_path = f'{input_db["bucket"]}/{input_db["formulas_chunks"]}/formula_to_id.pickle'
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

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=2048)
    futures = pw.map(build_ranking, [(formula_to_id_path, job_i, *job) for job_i, job in enumerate(ranking_jobs)])
    ranking_keys = [key for job_i, key in sorted(pw.get_result(futures))]
    pw.clean()

    rankings_df = pd.DataFrame(ranking_jobs, columns=['group_i', 'ranking_i', 'database_path', 'modifier', 'adduct'])
    rankings_df = rankings_df.assign(is_target=~rankings_df.adduct.isnull(), key=ranking_keys)

    return rankings_df


def calculate_fdrs(config, input_data, rankings_df):

    def run_ranking(key, data_stream, ibm_cos, decoy_bucket, decoy_key):
        target = pickle.loads(data_stream.read())
        decoy = pickle.loads(ibm_cos.get_object(Bucket=decoy_bucket, Key=decoy_key)['Body'].read())
        merged = pd.concat([target.assign(is_target=1), decoy.assign(is_target=0)])
        merged = merged.sort_values('msm', ascending=False)
        decoy_cumsum = (merged.is_target == False).cumsum()
        target_cumsum = (merged.is_target == True).cumsum()
        fdr = np.clip(decoy_cumsum / target_cumsum, 0, 1)
        fdr = pd.Series(np.minimum.accumulate(fdr[::-1])[::-1], merged.index)

        return merged.assign(fdr=fdr)[lambda df: df.is_target == 1].drop('is_target', axis=1)

    def merge_rankings(results):
        mols = pd.concat(results).rename_axis('formula_i').reset_index().groupby('formula_i')
        return mols.fdr.apply(np.median)

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=2048)

    # Run a separate map-reduce task for each independent set of rankings
    data_bucket = input_data['bucket']
    ranking_futures = []
    for group_i, group in rankings_df.groupby('group_i'):
        target_rows = group[group.is_target]
        decoy_rows = group[~group.is_target]

        for i, target_row in target_rows.iterrows():
            iterdata = [(f'{data_bucket}/{target_row.key}', data_bucket, decoy_row.key)
                        for i, decoy_row in decoy_rows.iterrows()]
            futures = pw.map_reduce(run_ranking, iterdata, merge_rankings)
            ranking_futures.append((target_row, futures))

    # Get & concat results
    fdr_dfs = []
    for target_row, futures in ranking_futures:
        results = pw.get_result(futures)
        print(results)
        result_df = pd.DataFrame({'database_path': target_row.database_path,
                                  'modifier': target_row.modifier,
                                  'adduct': target_row.adduct,
                                  'fdr': results},
                                 index=results.index)
        fdr_dfs.append(result_df)

    pw.clean()

    return pd.concat(fdr_dfs)
