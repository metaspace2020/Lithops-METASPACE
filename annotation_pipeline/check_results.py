import numpy as np

from annotation_pipeline.utils import logger


def get_reference_results(metaspace_options, ds_id):
    from metaspace.sm_annotation_utils import SMInstance
    if metaspace_options.get('host'):
        sm = SMInstance(host=metaspace_options['host'])
    else:
        sm = SMInstance()
    if metaspace_options.get('password'):
        sm.login(metaspace_options['email'], metaspace_options['password'])

    ds = sm.dataset(id=ds_id)
    reference_results = (ds.results('HMDB-v4')
        .reset_index()
        .rename({'moc': 'chaos', 'rhoSpatial': 'spatial', 'rhoSpectral': 'spectral'}, axis=1))
    return reference_results[['formula', 'adduct', 'chaos', 'spatial', 'spectral', 'msm', 'fdr']]


def check_results(results_df, ref_results):
    def quantize_fdr(fdr):
        if fdr <= 0.05: return 1
        if fdr <= 0.1: return 2
        if fdr <= 0.2: return 3
        if fdr <= 0.5: return 4
        return 5

    def find_differing_rows(df, col_a, col_b, max_error=0.001):
        return (df.assign(error=np.abs(df[col_a] - df[col_b]))
            .sort_values('error', ascending=False)
        [lambda d: d.error > max_error]
        [[col_a, col_b, 'error']])

    # clean up dataframes for better manual analysis & include only data that should be present in both dataframes
    filtered_results = (results_df
                        .rename({'mol': 'formula'}, axis=1)
                        [lambda df: (df.database_path == 'metabolomics/db/mol_db1.pickle')
                                    & (df.adduct != '')
                                    & (df.modifier == '')]
                        [['formula', 'adduct', 'chaos', 'spatial', 'spectral', 'msm', 'fdr']])

    merged_results = (ref_results
                      .merge(filtered_results, how='outer',
                             left_on=['formula', 'adduct'], right_on=['formula', 'adduct'], suffixes=['_ref', ''])
                      .sort_values(['formula', 'adduct']))

    # Validate no missing results (Should be zero as it's not affected by numeric instability)
    missing_results = merged_results[merged_results.fdr.isna() & merged_results.fdr_ref.notna()]

    # Find missing/wrong results for metrics & MSM
    common_results = merged_results[merged_results.fdr.notna() & merged_results.fdr_ref.notna()]
    spatial_wrong = find_differing_rows(common_results, 'spatial', 'spatial_ref')
    spectral_wrong = find_differing_rows(common_results, 'spectral', 'spectral_ref')
    chaos_wrong = find_differing_rows(common_results, 'chaos', 'chaos_ref')
    msm_wrong = find_differing_rows(common_results, 'msm', 'msm_ref')

    # Validate FDR (Results often go up/down one level due to random factor)
    fdr_level = merged_results.fdr.apply(quantize_fdr)
    fdr_ref_level = merged_results.fdr_ref.apply(quantize_fdr)

    fdr_exact = merged_results[fdr_level == fdr_ref_level]
    fdr_close = merged_results[np.abs(fdr_level - fdr_ref_level) <= 1]
    return {
        'merged_results': merged_results,
        'missing_results': missing_results,
        'spatial_wrong': spatial_wrong,
        'spectral_wrong': spectral_wrong,
        'chaos_wrong': chaos_wrong,
        'msm_wrong': msm_wrong,
        'fdr_exact': fdr_exact,
        'fdr_close': fdr_close,
    }



def log_bad_results(merged_results, missing_results, spatial_wrong, spectral_wrong, chaos_wrong, msm_wrong, fdr_exact, fdr_close):
    if len(missing_results):
        logger.error(f'{len(missing_results)} missing annotations: \n{missing_results.head()}')
    if len(spatial_wrong) > 5:
        # A small number of results are off by up to 1% due to an algorithm change since they were processed
        # Annotations with fewer than 4 ion images now have slightly higher spectral score than before
        logger.error(f'{len(spatial_wrong)} annotations with incorrect spatial metric:\n{spatial_wrong.head()}')
    if len(spectral_wrong):
        logger.error(f'{len(spectral_wrong)} annotations with incorrect spectral metric:\n{spectral_wrong.head()}')
    if len(chaos_wrong):
        logger.error(f'{len(chaos_wrong)} annotations with incorrect chaos metric:\n{chaos_wrong.head()}')
    if len(msm_wrong):
        logger.error(f'{len(msm_wrong)} annotations with incorrect MSM:\n{msm_wrong.head()}')
    if len(fdr_exact) < len(merged_results) * 0.75:
        logger.error(f'Not enough annotations with matching FDR: {len(fdr_exact)} out of {len(merged_results)}')
    if len(fdr_close) < len(merged_results) * 0.9:
        logger.error(f'Not enough annotations with FDR within tolerance: {len(fdr_close)} out of {len(merged_results)}')
