# Serverless METASPACE with Lithops
This repository demonstrates using [Lithops](https://github.com/lithops-cloud/lithops) to run the 
[METASPACE metabolite annotation pipeline](https://github.com/metaspace2020/metaspace/tree/master/metaspace/engine) on cloud resources.

METASPACE is a cloud engine for spatial metabolomics that performs molecular annotation of imaging mass spectrometry data. It takes an imaging mass spectrometry dataset and outputs molecules (e.g. metabolites and lipids) which are represented in the dataset, with assigned scores and false discovery rates. METASPACE is free and open source and is developed by the [Alexandrov team at EMBL Heidelberg](https://www.embl.de/research/units/scb/alexandrov/) with the generous European and US funding. It is used by a growing community over 500 users across the world. For more information, visit [METASPACE website](https://metaspace2020.eu).

Annotating high-resolution imaging mass spectrometry data often requires multiple CPU-days 
and >100GB of temporary storage, often making it impractical to run on typical desktop computers. 
Lithops allows processing to be almost seamlessly offloaded to cloud compute resources, 
rapidly scaling up to use as much compute power is available in your cloud of choice 
(e.g. 1000 parallel invocations in IBM Cloud) during intensive stages of the pipeline, 
and scaling down during less parallelizable stages to minimize cost. 

This repository includes two variant implementations of the annotation pipeline, selectable through runtime configuration:

* A purely Serverless Functions implementation, which runs on [any cloud Lithops supports](https://github.com/lithops-cloud/lithops#lithops-multi-cloud) 
(including IBM Cloud, Google Cloud, AWS, Azure and on-premise Knative/OpenWhisk installations).
* A hybrid Serverless + VM implementation, which enables several pipeline stages to use more efficient but more memory-intensive algorithms on large cloud VMs.
This configuration is currently only supported with IBM Cloud and on-premise VMs.

# Instructions for use

## Prerequisites:

* Python 3.8.5
* An account with a [supported cloud provider](https://github.com/lithops-cloud/lithops#move-to-the-cloud) (if running on a cloud platform) 
* Jupyter Notebook or Jupyter Lab (if running the benchmark notebooks)

## 1. Installation

Clone and install this repository with the following commands:
    
```
git clone https://github.com/metaspace2020/Lithops-METASPACE.git
cd Lithops-METASPACE
pip install -e .
```

## 2. Lithops Configuration

The purely Serverless and Hybrid implementations have different platform requirements when running on cloud platforms.
In "localhost" mode (i.e. not using cloud resources), both implementations are supported.

#### Localhost mode

This is the default mode. If you don't have any existing Lithops configuration, no configuration is needed.
If you have an existing Lithops config file, change the following values:
```yaml
lithops:
  mode: "localhost"
  storage: "localhost"
  workers: # Leave this blank to auto-detect CPU count
```

#### Pure Serverless mode

Follow the [Lithops instructions](https://github.com/lithops-cloud/lithops/tree/master/config) to configure a Serverless
compute backend and a storage backend. Additionally, set the following values in the Lithops config:
```yaml
lithops:
  mode: "serverless"
  include_modules: ["annotation_pipeline"]
  data_limit: false
  
serverless:
  runtime: "metaspace2020/annotation-pipeline-runtime:1.0.0-ibmcf-python38"
```

#### Hybrid mode

Hybrid mode requires both a Standalone and a Serverless executor to be configured, sharing the same storage backend. 
Currently this combination is only possible with IBM Virtual Private Cloud, IBM Cloud Functions and IBM Cloud Object Storage.

Follow the [Lithops instructions](https://github.com/lithops-cloud/lithops/tree/master/config) to configure the 3 backends. 
Additionally, set the following values in the Lithops config:
```yaml
lithops:
  mode: "serverless"
  include_modules: ["annotation_pipeline"]
  data_limit: false
  
serverless:
  runtime: "metaspace2020/annotation-pipeline-runtime:1.0.0-ibmcf-python38"
  
standalone:
  runtime: "metaspace2020/annotation-pipeline-runtime:1.0.0-ibmcf-python38"
```

## 3. Running the pipeline

#### Running the example notebooks

Launch Jupyter Notebook and open this directory. The main notebook is 
[`annotation-pipeline-demo.ipynb`](annotation-pipeline-demo.ipynb), which allows you to run
through the whole pipeline, and see the results at each step.

There are also 3 notebooks prepared for benchmarking:

1. [`experiment-1-typical.ipynb`](./experiment-1-typical.ipynb) - Demonstrates running through the whole 
    Serverless metabolite annotation pipeline with a typical dataset,  
    downloading the results and comparing them against the Serverful implementation of METASPACE.
2. [`experiment-2-interactive.ipynb`](./experiment-2-interactive.ipynb) - An example of running the pipeline against 
    a smaller set of molecules, to demonstrate the potential of Serverless to provide low-latency access 
    to computating resources.
3. [`experiment-3-large.ipynb`](./experiment-3-large.ipynb) - A stress test that runs the Serverless metabolite 
    annotation pipeline with a large dataset and many molecular databases.

#### Running from the command line

```
usage: python3 -m annotation_pipeline annotate [ds_config.json] [db_config.json] [output path]

positional arguments:
  ds_config.json        ds_config.json path
  db_config.json        db_config.json path
  output                directory to write output files
optional arguments:
  -h, --help            show this help message and exit
  --no-output           prevents outputs from being written to file
  --no-cache            prevents loading cached data from previous runs
  --impl {serverless,hybrid,auto}
                        Selects whether to use the Serverless or Hybrid
                        implementation. "auto" will select the Hybrid
                        implementation if the selected platform is supported
                        and correctly configured (running in localhost mode,
                        or in serverless mode with ibm_vpc configured)
```

## Input data

The main inputs to the pipeline are specified in two JSON files: the dataset and database configs. 
There are example config files in the [`metabolomics`][metabolomics] directory.

#### Dataset configs
Dataset configs should follow this format:
```json5
{
  "name": "****",                                      // A unique name for this dataset (used for caching)
  "imzml_path": "https://****.imzML or C:\****.imzML", // URL or filesystem path to the .imzML file 
  "ibd_path": "https://****.ibd or C:\****.ibd",       // URL or filesystem path to the .ibd file
  "num_decoys": 20,                                    // Number of decoys to use for FDR calculation (can be any integer between 1 and 80) 
  "polarity": "+",                                     // Ionization mode of the dataset ("+" or "-")
  "isocalc_sigma": 0.001238,                           // The "sigma" parameter representing the expected peak width at 200 m/z based on the instrument's resolving power
                                                       // Common values are:
                                                       // RP 70,000 @ 200 m/z: 0.002476
                                                       // RP 140,000 @ 200 m/z: 0.001238
                                                       // RP 200,000 @ 200 m/z: 0.000867
                                                       // RP 280,000 @ 200 m/z: 0.000619
  "metaspace_id": "**** (Optional)"                    // Optional ID of a dataset at https://metaspace2020.eu to validate the results against
}
```

The imzML and ibd files may also be specified as URL-like paths to cloud storage, 
e.g. `cos://datasets/ds.imzML` for IBM COS or `s3://datasets/ds.imzML` for AWS S3.

#### Database configs

Database configs should follow this format:
```json5
{
  "name": "db_configN",                                // A unique name for this database (used for caching)
  "databases": ["metabolomics/db/mol_db1.csv"],        // Filesystem path to CSV file containing formulas
  "adducts": ["","+H","+Na","+K"],                     // Adducts to search for
  "modifiers": ["", "-H2O", "-CO2", "-NH3"]            // Neutral losses or chemical modifications to search for
}
```
    
### Example datasets

| Dataset                             | Author                            | Config file |
| :---------------------------------: | :-------------------------------: | :---------: |
| [Brain02_Bregma1-42_02](https://metaspace2020.eu/annotations?ds=2016-09-22_11h16m11s) | Régis Lavigne,<br/>University of Rennes 1 | `ds_config1.json` | 
| [AZ_Rat_Brains](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m53s) | Nicole Strittmatter,<br/>AstraZeneca | `ds_config2.json` | 
| [CT26_xenograft](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m49s) | Nicole Strittmatter,<br/>AstraZeneca | `ds_config3.json` | 
| [Mouse brain test434x902](https://metaspace2020.eu/annotations?ds=2019-07-31_17h35m11s) <br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen | `ds_config4.json` | 
| [X089-Mousebrain_842x603](https://metaspace2020.eu/annotations?ds=2019-08-19_11h28m42s) <br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen | `ds_config5.json` | 
| Microbial interaction slide | Don Nguyen,<br/>European Molecular Biology Laboratory | `ds_config6.json` | 

### Example databases

These molecular databases can be selected in the `ds_config.json` files. They are automatically converted to 
pickle format and uploaded to IBM cloud in the notebooks. 

| Database            | Filename            | Description                    |
| :-----------------: | :-----------------: | :----------------------------- |
| [HMDB](http://www.hmdb.ca/) | `mol_db1.csv` | Human Metabolome Database |
| [ChEBI](https://www.ebi.ac.uk/chebi/) | `mol_db2.csv` | Chemical Entities of Biological Interest |
| [LIPID MAPS](https://www.lipidmaps.org/) | `mol_db3.csv` |  |
| [SwissLipids](https://www.swisslipids.org/) | `mol_db4.csv` |  |
| Small database | `mol_db5.csv` | This database is used in Experiment 2 as an example of a small set of user-supplied molecules for running small, interactive annotation jobs. |
| Peptide databases | `mol_db7.csv` <br/> ... <br/> `mol_db12.csv` | A collection of databases of predicted peptides. These databases were contributed by [Benjamin Baluff (M4I, Maastricht University)](https://www.maastrichtuniversity.nl/b.balluff) exclusively for use with METASPACE. |  

# Acknowledgements

![image](https://user-images.githubusercontent.com/26366936/61350554-d62acf00-a85f-11e9-84b2-36312a35398e.png)

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 825184.
