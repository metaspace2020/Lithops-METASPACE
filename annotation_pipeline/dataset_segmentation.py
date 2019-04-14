import pickle
import sys
import numpy as np
import pywren_ibm_cloud as pywren
from .dataset import parse_txt, parse_spectrum_line, reduce_chunks


def generate_segm_intervals(config, input_db, segm_n):
    def get_segm_intervals(key, data_stream):
        centroids_df = pickle.loads(data_stream.read())
        segm_bounds_q = [i * 1 / segm_n for i in range(1, segm_n)]
        segm_bounds = [np.quantile(centroids_df.mz.values, q) for q in segm_bounds_q]
        segm_bounds = [0.] + segm_bounds + [sys.float_info.max]
        segm_intervals = list(zip(segm_bounds[:-1], segm_bounds[1:]))
        return segm_intervals

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=512)
    iterdata = [f'{input_db["bucket"]}/{input_db["centroids_pandas"]}']
    pw.map(get_segm_intervals, iterdata)
    segm_intervals = pw.get_result()
    pw.clean()

    return segm_intervals


def split_spectra_into_segments(config, input_data, segm_n, segm_intervals):
    def iterate_over_segment(key, data_stream, min_mz, max_mz):
        spectra = parse_txt(key, data_stream, parse_spectrum_line)
        rows = []
        for sp_i, mzs, ints in spectra:
            smask = (mzs >= min_mz) & (mzs <= max_mz)
            rows.append(([sp_i], mzs[smask], ints[smask]))
        return rows

    def store_segm(key, data_stream, ibm_cos, segm_i, interval):
        pw = pywren.ibm_cf_executor(config=config, runtime_memory=512)
        iterdata = [[f'{input_data["bucket"]}/{input_data["ds"]}', *interval]]
        pw.map_reduce(iterate_over_segment, iterdata, reduce_chunks, chunk_size=64*1024**2)
        segm_spectra = pickle.dumps(np.array(pw.get_result()))
        ibm_cos.put_object(Bucket=input_data["bucket"], Key=f'{input_data["segments"]}/{segm_i}.pickle', Body=segm_spectra)

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=256)
    iterdata = [[f'{input_data["bucket"]}/{input_data["ds"]}', segm_i, segm_intervals[segm_i]] for segm_i in range(segm_n)]
    futures = pw.map(store_segm, iterdata)
    pw.get_result(futures)
    pw.clean()


def clean_segments(config, input_data):
    def clean_segments_datasets(bucket, key, data_stream, ibm_cos):
        ibm_cos.delete_object(Bucket=bucket, Key=key)

    pw = pywren.ibm_cf_executor(config=config, runtime_memory=256)
    data_stream = f'{input_data["bucket"]}/{input_data["segments"]}'
    pw.map(clean_segments_datasets, data_stream)
    pw.get_result()
    pw.clean()