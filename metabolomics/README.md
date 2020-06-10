These files hold sample configurations for datasets and databases for the annotation pipeline. 
Each run of the pipeline uses one db_config and one ds_config. 
The computational requirements of the pipeline scale with both the size of the database, and the size of the dataset.

## Dataset configurations

| Config file | Data size     | Dataset                             | Author                            | 
| :---------: | :-----------: | :---------------------------------: | :-------------------------------: | 
| `ds_config1.json` | 0.05 GB | [Brain02_Bregma1-42_02](https://metaspace2020.eu/annotations?ds=2016-09-22_11h16m11s) | Régis Lavigne,<br/>University of Rennes 1 |
| `ds_config2.json` | 0.7 GB  | [AZ_Rat_Brains](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m53s) | Nicole Strittmatter,<br/>AstraZeneca |
| `ds_config3.json` | 1.8 GB  | [CT26_xenograft](https://metaspace2020.eu/annotations?ds=2016-09-21_16h06m49s) | Nicole Strittmatter,<br/>AstraZeneca |
| `ds_config4.json` | 3.9 GB  | [Mouse brain test434x902](https://metaspace2020.eu/annotations?ds=2019-07-31_17h35m11s)<br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen |
| `ds_config5.json` | 7.0 GB  | [X089-Mousebrain_842x603](https://metaspace2020.eu/annotations?ds=2019-08-19_11h28m42s)<br/>Captured with AP-SMALDI5<br/> and Q Exactive HF Orbitrap | Dhaka Bhandari,<br/>Justus-Liebig-University Giessen |
| `ds_config6.json` | 56.7 GB | Microbial interaction slide | Don Nguyen,<br/>European Molecular Biology Laboratory |

## Database configurations

| Config file | Size (# formulas) |  
| :---------: | :---------------: |  
| `db_config1.json` | 12K  |
| `db_config2.json` | 29K  | 
| `db_config3.json` | 74K  | 
| `db_config4.json` | 229K | 
| `db_config5.json` | 957K | 
| `db_config6.json` | 2.2M | 
| `db_config7.json` | 6.0M | 

## Recommended configs

For comparing pipeline performance against Serverful METASPACE, it's most useful to have measurements 
where the DS and DB are varied separately. This can help identify which parts of the pipeline are more efficient
in each implementation:
* `ds_config2.json` / `db_config2.json` - Closest to an average METASPACE job
* `ds_config2.json` / `db_config3.json` 
* `ds_config5.json` / `db_config2.json` 
* `ds_config5.json` / `db_config3.json` 

For testing that extreme cases also successfully finish processing:
* `ds_config6.json` / `db_config2.json` - The largest real-world dataset processed by METASPACE
* `ds_config2.json` / `db_config5.json` - The largest real-world database processed by METASPACE

For quickly testing that the pipeline runs successfully:
* `ds_config1.json` / `db_config1.json`

For demonstrating scalability beyond the serverful implementation:
* Any dataset and `db_config6.json` or `db_config7.json`  