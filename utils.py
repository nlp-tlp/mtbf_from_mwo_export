'''
Utilities and connection functions

@author: Tyler Bikaun
'''

import yaml
import sys, traceback
import json
import pandas as pd

def load_config(path=None):
    """Loads configuration file from disk"""
    if path is None:
        path = r'config.yaml'

    try:
        with open(path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(e)
        
        
def json2csv(json_path: str, csv_path: str, cost_or_hours: str):
    ''' Converts JSON file to CSV for improved readability of model results'''
    
    with open(json_path, 'r') as fjr:
        data = json.load(fjr)

    # Create FLOC: MTTF pairs
    floc_mttf_data = {'FLOC': [], 'MTTF': [], 'ETA': [], 'BETA': [], 'COUNTS': [], 'WO_FS_DESC': [], 'WO_FS_TIME': [], f'WO_FS_{cost_or_hours.upper()}': [], "WO_FS_CLF": [], "WO_FS_FOS": [],
                    'WO_NFS_DESC': [], f'WO_NFS_{cost_or_hours.upper()}': [], "WO_NFS_CLF": []}
    
    data = data["RESULTS"] if "RESULTS" in list(data.keys()) else data
    
    for result in data:
        
        floc = result
        mttf = data[result]["MTTF"] 
        eta = data[result]["ETA"]
        beta = data[result]["BETA"]
        counts = "\n".join([f'{name}: {value}' for name, value in data[result]["COUNTS"].items()])
        wo_fs_desc = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTIONS_FS"]["WO_DESCRIPTION"].items()])
        wo_fs_time = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTIONS_FS"]["TIME"].items()])
        wo_fs_cost = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTIONS_FS"][("TOTAL_ACTUAL_COST" if cost_or_hours == 'cost' else "TOTAL_ACTUAL_HOURS")].items()])
        wo_fs_clf = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTIONS_FS"]["WO_CLASSIFICATION"].items()])
        wo_fs_fos = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTIONS_FS"]["FAILURE_OR_SUSPENSION"].items()])

        wo_nfs_desc = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTION_NFS"]["WO_DESCRIPTION"].items()])
        wo_nfs_cost = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTION_NFS"][("TOTAL_ACTUAL_COST" if cost_or_hours == 'cost' else "TOTAL_ACTUAL_HOURS")].items()])
        wo_nfs_clf = "\n".join([f'{value}' for name, value in data[result]["WO_DESCRIPTION_NFS"]["WO_CLASSIFICATION"].items()])
        
        floc_mttf_data['FLOC'].append(floc)
        floc_mttf_data['MTTF'].append(mttf)
        floc_mttf_data['ETA'].append(eta)
        floc_mttf_data['BETA'].append(beta)
        floc_mttf_data['COUNTS'].append(counts)
        floc_mttf_data['WO_FS_DESC'].append(wo_fs_desc)
        floc_mttf_data['WO_FS_TIME'].append(wo_fs_time)
        floc_mttf_data[f'WO_FS_{cost_or_hours.upper()}'].append(wo_fs_cost)
        floc_mttf_data['WO_FS_CLF'].append(wo_fs_clf)
        floc_mttf_data['WO_FS_FOS'].append(wo_fs_fos)
        floc_mttf_data['WO_NFS_DESC'].append(wo_nfs_desc)
        floc_mttf_data[f'WO_NFS_{cost_or_hours.upper()}'].append(wo_nfs_cost)
        floc_mttf_data['WO_NFS_CLF'].append(wo_nfs_clf)
        
    df = pd.DataFrame(floc_mttf_data)
    df.to_csv(csv_path, index = False)

if __name__ == '__main__':
    mttf_fname_json = r'exp/A/results collocation/eval_A_S3.json'
    mttf_fname_csv = f'{mttf_fname_json.split(".")[0]}.csv'
    
    # If no structured fields used, default to 'hours'
    json2csv(json_path = mttf_fname_json, csv_path = mttf_fname_csv, cost_or_hours = 'cost')