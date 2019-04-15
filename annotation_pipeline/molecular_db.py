import pywren_ibm_cloud as pywren
import pandas as pd
import pickle


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
