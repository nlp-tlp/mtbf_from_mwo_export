# Semi-automated Estimation of Reliability Measures from Maintenance Work Order Records
## Overview
This repository contains associated code for conference paper Semi-Automated Estimation of Reliability Measures from Maintenance Work Order Records. 

![pipeline image](https://code-ittc.csiro.au/tyler.bikaun/mtbf_from_mwo/-/blob/master/model_overview.png)


## Installation
```
    $ pip install requirements.txt
```

## Data
To use the pipeline within this repository, an extract of maintenance work order records from 1SAP, SAP or JDE is required with the suitable fields available that map to the following:

- FUNCTIONAL_LOC_DESC - e.g. "Functional Location Description"
- FUNCTIONAL_LOC - e.g. "Work Order Functional Location"
- WO_DESCRIPTION - e.g. "Work Order Description"
- CREATION_DATE - e.g. "Work Order Created On Timestamp (UTC)"
- BASIC_START_DATE - e.g. "Work Order Basic Start Date"
- BASIC_FINISH_DATE - e.g. "Work Order Basic Finish Date"
- ACTUAL_START_DATE - e.g. "Work Order Actual Start Timestamp"
- ACTUAL_FINISH_DATE - e.g. "Work Order Actual Finish Timestamp"
- TOTAL_ACTUAL_HOURS - e.g. "Work Order Actual Total Hours"
- TOTAL_ACTUAL_COST - e.g. "Work Order Total Actual Costs"
- WO_CLASSIFICATION - e.g. "Work Order Type"
	- Classification is either WO TYPE in SAP or Classification in JDE (preventative, corrective, etc.)
- OBJECT_TYPE - eg. "Functional Location Object Type Description"
	- This is a high level object type (if avaialble) that identifies clearly what type of object (e.g. pump, motor, conveyor) the work order is associated with.
- ACTIVITY_TYPE - JDE specific - this has repair, overhaul, rebuild etc, classifications for work orders.


## Usage
Before executing any code, a configuration file in YAML (.yml) format is required (a template for the configuration file is provided in `config_template.yml with details about field value requirements).

Once a suitable data set has been selected and the configuration file set, the pipeline can be executed with `python run.py`. `run.py` will execute the following stages of the pipeline:
1. Term expansion process via `term_expansion.py`
2. Collocation identification via `collocations.py`
3. Parameter estimation via `mwo_to_mtbf.py`



## Attribution
Please cite our [[conference paper]](https://arxiv.org/abs/####.#####) (to appear in EPHM 2021) if you find it useful in your research:
```
  @inproceedings{bikaun2021semiauto,
      title={Semi-automated Estimation of Reliability Measures from Maintenance Work Order Records},
      author={Bikaun, Tyler, and Hodkiewicz, Melinda},
      journal={European Conference of the Prognostics and Health Management Society, PHM Society'2021.},
      pages={x--y},
      year={2021}
}
```

## Contact
Please email any questions or queries to Tyler Bikaun (tyler.bikaun@research.uwa.edu.au)