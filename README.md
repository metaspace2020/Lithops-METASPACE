# METASPACE annotation pipeline on IBM Cloud
Experimental code to integrate [METASPACE engine](https://github.com/metaspace2020/metaspace/tree/master/metaspace/engine)
with [PyWren](https://github.com/pywren/pywren-ibm-cloud) for IBM Cloud.

# Instructions for use

### Prerequisites:
* **Python 3.7.x**

    Python must be one of the 3.7 versions (i.e. not 3.8 or above, not 3.6 or below) to work with the pre-built runtime. 

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

2. Copy `config.json.template` to `config.json` and edit it, filling in your IBM Cloud details. It is fine to use the same bucket in all places. 

3. Run one of the below notebooks. 

### Example notebooks

The main notebook is [`pywren-annotation-pipeline-demo.ipynb`](pywren-annotation-pipeline-demo.ipynb), which allows you to run
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
    
### Example datasets

| Dataset                             | Author                            | Config file |
| :---------------------------------: | :-------------------------------: | :---------: |
| [Brain02_Bregma1-42_02](https://metaspace2020.eu/annotations?ds=2016-09-22_11h16m11s) | Régis Lavigne,<br/>University of Rennes 1 | `ds_config1.json` | 
| [AZ_Rat_Brains](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m53s) | Nicole Strittmatter,<br/>AstraZeneca | `ds_config2.json` | 
| [CT26_xenograft](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m49s) | Nicole Strittmatter,<br/>AstraZeneca | `ds_config3.json` | 
| [Mouse brain test434x902](https://metaspace2020.eu/annotations?ds=2019-07-31_17h35m11s)<br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen | `ds_config4.json` | 
| [X089-Mousebrain_842x603](https://metaspace2020.eu/annotations?ds=2019-08-19_11h28m42s)<br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen | `ds_config5.json` | 
| Microbial interaction slide | Don Nguyen,<br/>European Molecular Biology Laboratory | `ds_config6.json` | 

### Example databases

These molecular databases can be selected in the `ds_config.json` files. They are automatically converted to 
pickle format and uploaded to IBM cloud in the notebooks. 

| Database            | Filename            | Description                    |
| :-----------------: | :-----------------: | :----------------------------- |
| [HMDB](http://www.hmdb.ca/) | `mol_db1.pickle` | Human Metabolome Database |
| [ChEBI](https://www.ebi.ac.uk/chebi/) | `mol_db2.pickle` | Chemical Entities of Biological Interest |
| [LIPID MAPS](https://www.lipidmaps.org/) | `mol_db3.pickle` |  |
| [SwissLipids](https://www.swisslipids.org/) | `mol_db4.pickle` |  |
| Small database | `mol_db5.pickle` | This database is used in Experiment 2 as an example of a small set of user-supplied molecules for running small, interactive annotation jobs. |
| Peptide databases | `mol_db7.pickle` <br/> ... <br/> `mol_db12.pickle` | A collection of databases of predicted peptides. These databases were contributed by [Benjamin Baluff (M4I, Maastricht University)](https://www.maastrichtuniversity.nl/b.balluff) exclusively for use with METASPACE. |  

# Acknowledgements

![image](https://user-images.githubusercontent.com/26366936/61350554-d62acf00-a85f-11e9-84b2-36312a35398e.png)

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 825184.
