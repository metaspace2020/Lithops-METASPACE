# Metaspace annotation pipeline on IBM Cloud
Experimental code to integrate Metaspace [engine](https://github.com/metaspace2020/metaspace/tree/master/metaspace/engine)
with [PyWren](https://github.com/pywren/pywren-ibm-cloud) for IBM Cloud.

# Instructions for use

### Prerequisites:
* **Python 3.6.x**

    Python must be one of the 3.6 versions (i.e. not 3.7 or above, not 3.5 or below) to work with the pre-built runtime. 

* **IBM Cloud account**

    1. Sign up here: https://cloud.ibm.com/
    2. Create a Cloud Object Storage bucket
    3. Create a IBM Cloud Functions namespace and CloudFoundry organization, ideally in the same region as the Cloud Object Storage bucket.

* **Jupyter Notebook or Jupyter Lab**

### Setup

1. Clone and install this repository with the following commands:
    
    ```
    git clone https://github.com/metaspace2020/pywren-annotation-pipeline.git
    cd pywren-annotation-pipeline
    pip install -e .
    ```

2. Download the [sample data](https://s3.eu-de.cloud-object-storage.appdomain.cloud/pywren-annotation-pipeline-public/metabolomics.tar.gz) and extract it into 
    this directory, merging it with the existing files. Verify that the directory structure has merged correctly by checking that the file `metabolomics/ds/CT26_xenograft/Image1_CT26.imzML` exists. 

3. Copy `config.json.template` to `config.json` and edit it, filling in your IBM Cloud details. It is fine to use the same bucket in all places. 

### Example notebooks

The main notebook is [`pywren-annotation-pipeline.ipynb`](./pywren-annotation-pipeline.ipynb), which allows you to run
through the whole pipeline, and see the results at each step.

There are also 3 notebooks prepared for benchmarking that can be run with Jupyter Notebook:

1. [`experiment-1-typical.ipynb`](./experiment-1-typical.ipynb) - Demonstrates running through the whole 
    Serverless metabolite annotation pipeline with a typical dataset,  
    downloading the results and comparing them against the Serverful implementation of METASPACE.
2. [`experiment-2-interactive.ipynb`](./experiment-2-interactive.ipynb) - An example of running the pipeline against 
    a smaller set of molecules, to demonstrate the potential of Serverless to provide low-latency access 
    to computating resources.
3. [`experiment-3-large.ipynb`](./experiment-3-large.ipynb) - A stress test that runs the Serverless metabolite 
    annotation pipeline with a large dataset and many molecular databases.
    
# Acknowledgements

![image](https://user-images.githubusercontent.com/26366936/61350554-d62acf00-a85f-11e9-84b2-36312a35398e.png)

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 825184.
