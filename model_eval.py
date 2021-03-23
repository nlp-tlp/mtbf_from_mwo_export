"""
Script for randomly selecting data from MTTF output for human evaluation.

@author: Tyler Bikaun
"""

import json
import random

# Parameters
output_prefix = 'ALCOA_'    # options: ALCOA_ or NKW_
samples_no = 20

fname_data = 'exp/A/results collocation/mttf_output_A_S3.json'   #f'data/outputs/{output_prefix}mttf_output.json'
fname_eval = 'exp/A/results collocation/eval_A_S3.json'      #f'data/outputs/{output_prefix}eval.json'

# Load data
with open(fname_data, 'r') as fr:
    data = json.load(fr)
    
data = data['RESULTS']

# Randomly sample data
sample_keys = random.sample(data.keys(), samples_no)

# Save data
output_data = {}
for key in sample_keys:
    output_data[key] = data[key]
    
with open(fname_eval, 'w') as fw:
    json.dump(output_data, fw)