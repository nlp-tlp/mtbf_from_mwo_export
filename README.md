# Semi-automated Estimation of Reliability Measures from Maintenance Work Order Records
## Overview
This repository contains implementations anda associated code for Semi-automated Estimation of Reliability Measures from Maintenance Work Order Records. 

![pipeline image](https://github.com/uwasystemhealth/mwo_to_mttf/blob/main/model_overview.png)


### Installation
Please install the provided requirements:
```
    $ pip install requirements.txt
```

### Datasets
...

### Usage
Before executing code ..., firstly place any maintenance work order records in .csv format into the /data directory.
## Directory Structure
    mwo_to_mttf
    └──  data
            └──  outputs
                    └──  PM02_for_qa.csv
                    └──   fs_data.xlsx
                    └──  mttf.csv
            └──  token - Final 2.xlsx
            └──  seed_terms_replace.xlsx
            └──  ngrams_pumps.xlsx
            └──  new_terms_replace.xlsx
    └──  MTTF Calculator.ipynb
    └──  Pump Dataset - term generator.ipynb
    └──  README.md

Please cite our [[conference paper]](https://arxiv.org/abs/####.#####) (to appear in #### 202#) if you find it useful in your research:
```
  @article{bikaun2021semiauto,
      title={Semi-automated Estimation of Reliability Measures from Maintenance Work Order Records},
      author={Bikaun, Tyler, and Hodkiewicz, Melinda},
      year={2021},
      journal={arXiv preprint arXiv:####.#####},
      eprint={####.#####},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Contact
Please email any questions or queries to Tyler Bikaun (tyler.bikaun@research.uwa.edu.au)