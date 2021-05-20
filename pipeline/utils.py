'''
Utilities and connection functions for pipeline.

@author: Tyler Bikaun
'''

import yaml
import json
import pandas as pd


def load_config(path: str = None):
  '''
    Loads configuration file from disk.

    Parameters
    ----------
    path: Str
        Path to YAML configuration file on disk
    Returns
    -------
    config: Dict
        YAML configuration object as Python dictionary
  '''

  if path is None:
    path = 'config_template.yaml'
  try:
    with open(path) as file:
      config = yaml.load(file, Loader=yaml.FullLoader)
    return config
  except Exception as e:
    print(e)


def json2csv(json_path: str, csv_path: str, cost_or_hours: str = 'hours'):
  '''
    Converts pipeline output from JSON to CSV for improved readability of results.

    Parameters
    ----------
    json_path: Str
        Path to results in JSON format
    csv_path: Str
        Desired path to save results in CSV format
    cost_or_hours: Str
        Structured field used in configuration file to discriminate detected EOL events

  '''

  assert str(json_path).split('.')[-1] == 'json'
  assert str(csv_path).split('.')[-1] == 'csv'

  with open(json_path, 'r') as fjr:
    data = json.load(fjr)

  # Create functional location MTBF pairs
  floc_mtbf_data = {
      'FLOC': [],
      'MTBF': [],
      'ETA': [],
      'BETA': [],
      'COUNTS': [],
      'WO_FS_DESC': [],
      'WO_FS_TIME': [],
      f'WO_FS_{cost_or_hours.upper()}': [],
      'WO_FS_CLF': [],
      'WO_FS_FOS': [],
      'WO_NFS_DESC': [],
      f'WO_NFS_{cost_or_hours.upper()}': [],
      'WO_NFS_CLF': []
  }

  data = data['RESULTS'] if 'RESULTS' in list(data.keys()) else data

  # Iterate over data and create structured records
  for result in data:
    floc = result
    mtbf = data[result]['MTBF']
    eta = data[result]['ETA']
    beta = data[result]['BETA']
    counts = '\n'.join(
        [f'{name}: {value}' for name, value in data[result]['COUNTS'].items()])
    wo_fs_desc = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTIONS_FS']
        ['WO_DESCRIPTION'].items()
    ])
    wo_fs_time = '\n'.join([
        f'{value}'
        for name, value in data[result]['WO_DESCRIPTIONS_FS']['TIME'].items()
    ])
    wo_fs_cost = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTIONS_FS'][(
            'TOTAL_ACTUAL_COST' if cost_or_hours ==
            'cost' else 'TOTAL_ACTUAL_HOURS')].items()
    ])
    wo_fs_clf = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTIONS_FS']
        ['WO_CLASSIFICATION'].items()
    ])
    wo_fs_fos = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTIONS_FS']
        ['FAILURE_OR_SUSPENSION'].items()
    ])

    wo_nfs_desc = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTION_NFS']
        ['WO_DESCRIPTION'].items()
    ])
    wo_nfs_cost = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTION_NFS'][(
            'TOTAL_ACTUAL_COST' if cost_or_hours ==
            'cost' else 'TOTAL_ACTUAL_HOURS')].items()
    ])
    wo_nfs_clf = '\n'.join([
        f'{value}' for name, value in data[result]['WO_DESCRIPTION_NFS']
        ['WO_CLASSIFICATION'].items()
    ])

    floc_mtbf_data['FLOC'].append(floc)
    floc_mtbf_data['MTBF'].append(mtbf)
    floc_mtbf_data['ETA'].append(eta)
    floc_mtbf_data['BETA'].append(beta)
    floc_mtbf_data['COUNTS'].append(counts)
    floc_mtbf_data['WO_FS_DESC'].append(wo_fs_desc)
    floc_mtbf_data['WO_FS_TIME'].append(wo_fs_time)
    floc_mtbf_data[f'WO_FS_{cost_or_hours.upper()}'].append(wo_fs_cost)
    floc_mtbf_data['WO_FS_CLF'].append(wo_fs_clf)
    floc_mtbf_data['WO_FS_FOS'].append(wo_fs_fos)
    floc_mtbf_data['WO_NFS_DESC'].append(wo_nfs_desc)
    floc_mtbf_data[f'WO_NFS_{cost_or_hours.upper()}'].append(wo_nfs_cost)
    floc_mtbf_data['WO_NFS_CLF'].append(wo_nfs_clf)

  df = pd.DataFrame(floc_mtbf_data)
  df.to_csv(csv_path, index=False)


if __name__ == '__main__':
  mtbf_fname_json = 'results.json'
  mtbf_fname_csv = f"{mtbf_fname_json.split('.')[0]}.csv"

  json2csv(json_path=mtbf_fname_json,
           csv_path=mtbf_fname_csv,
           cost_or_hours='cost')
