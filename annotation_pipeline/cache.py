import pickle
from pywren_ibm_cloud.storage.utils import CloudObject

from annotation_pipeline.utils import get_ibm_cos_client, list_keys, clean_from_cos


class PipelineCacher:
    def __init__(self, pw, dataset_name):
        self.pywren_executor = pw
        self.config = self.pywren_executor.config

        self.storage_handler = get_ibm_cos_client(self.config)
        self.bucket = self.config['pywren']['storage_bucket']
        self.prefix = f'metabolomics/cache/{dataset_name}'

    def load(self, key):
        data_stream = self.storage_handler.get_object(Bucket=self.bucket, Key=key)['Body']
        return pickle.loads(data_stream.read())

    def save(self, data, key):
        p = pickle.dumps(data)
        self.storage_handler.put_object(Bucket=self.bucket, Key=key, Body=p)

    def exists(self, key):
        try:
            self.storage_handler.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def clean(self):
        keys = list_keys(self.bucket, self.prefix, self.storage_handler)

        cobjects_to_clean = []
        for cache_key in keys:
            cache_data = self.load(cache_key)
            if isinstance(cache_data, tuple):
                for obj in cache_data:
                    if isinstance(obj, list):
                        if isinstance(obj[0], CloudObject):
                            cobjects_to_clean.extend(obj)
                    elif isinstance(obj, CloudObject):
                        cobjects_to_clean.append(obj)
            elif isinstance(cache_data, list):
                if isinstance(cache_data[0], CloudObject):
                    cobjects_to_clean.extend(cache_data)
            elif isinstance(cache_data, CloudObject):
                cobjects_to_clean.append(cache_data)

        self.pywren_executor.clean(cs=cobjects_to_clean)
        clean_from_cos(self.config, self.bucket, self.prefix, self.storage_handler)
