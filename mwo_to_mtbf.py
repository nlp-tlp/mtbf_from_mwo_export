''' 
Script for standardised and scalable conversion of maintenance work orders to MTBF estimates

@authors: Tyler Bikaun & Melinda Hodkiewicz
'''

# Import libraries
import yaml
import json
import math
import pandas as pd
import numpy as np
from datetime import date, datetime
import reliability as rb
import weibull as wb
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as st
from math import sqrt
import re
import time
from pathlib import Path

from utils import load_config, json2csv


class DataLoader:
    """ Token and work order data loader """
    def __init__(self, config):
        self.config = config
        self.workorder_path = config['File']['workorderPath']
        self.workorder_col_names = config['Data']['workorderColNames']

        self.create_token_lists()
        self.load_workorders()

    def load_tokens(self):
        """ Loads SME defined tokens / term data from Gazetteer 
            Note: format of token data must be col ['Term', 'Action'] """
            
        self.token_data = pd.read_excel(Path(self.config['File']['outputDir']) / 'token_file.xlsx')
        self.token_data = self.token_data[['Term', 'Action']]
        
        # Lower-case terms so that they are easier to do sub-string matching against
        self.token_data['Term'] = self.token_data['Term'].apply(lambda x: str(x).lower())
        
        print(f'{datetime.now()}: Loaded tokens from disk.')

    def create_token_lists(self):
        """ Creates token lists for repair and replacement events """
        
        self.load_tokens()
        
        try:
            self.repair_selection = self.token_data[self.token_data['Action'] == '_Repair']['Term'].tolist()
        except Exception as e:
            print(f'{datetime.now()}: Error generating repair selection list - {e}')

        try:
            self.replace_selection = self.token_data[self.token_data['Action'] == '_Replace']['Term'].tolist()
        except Exception as e:
            print(f'{datetime.now()}: Error generating replace selection list - {e}')
            
        # If there is difference between repair and replace selections, default goes to repair
        self.replace_selection = list(set(self.replace_selection).difference(set(self.repair_selection)))

    def load_workorders(self):
        """ Loads workorder data from disk """
        
        if self.workorder_path.split('.')[-1] == 'xlsb':
            # Note: required pyxlsb library (pip install pyxlsb)
            self.mntn_data_all = pd.read_excel(self.workorder_path, engine = 'pyxlsb')

            # Convert int date to datetime objects
            # Note: Reference date = 693594 -> https://stackoverflow.com/questions/6706231/fetching-datetime-from-float-in-python
            date_cols = [self.workorder_col_names['woCreationDate'],
                        self.workorder_col_names['woBasicStartDate'],
                        self.workorder_col_names['woBasicFinishDate'],
                        self.workorder_col_names['woActualStartDate'],
                        self.workorder_col_names['woActualFinishDate']]
            for col in date_cols:
                self.mntn_data_all[col] = self.mntn_data_all[col].apply(lambda x: date.fromordinal(x + 693594))

        if self.workorder_path.split('.')[-1] == 'xlsx':
            self.mntn_data_all = pd.read_excel(self.workorder_path)

        if self.workorder_path.split('.')[-1] == 'csv':
            self.mntn_data_all = pd.read_csv(self.workorder_path, encoding = "ISO-8859-1")


class DataWrangler(DataLoader):
    """ Wrangler for work order data """
    def __init__(self, config):
        DataLoader.__init__(self, config)

        self.filters = config['Data']['filters']
        self.cmms_type = config['Data']['CMMSType']
        
        self.data_preprocess()
        self.return_to_function()
        self.token_match()
        self.identify_fs()

    def data_preprocess(self):
        """ Preprocess data - name normalisation, data type changes, basic filtering etc. """

        start_size = len(self.mntn_data_all)

        for field in self.workorder_col_names:
            if field == 'additionalCols':
                pass
            elif self.workorder_col_names[field]:
                self.mntn_data_all.rename(columns = {self.workorder_col_names[field] : field}, inplace = True)

        # Subset columns of interest (plus additional ones if required)
        col_subset = []
        for field in self.workorder_col_names:
            if field == 'additionalCols':
                pass
            elif self.workorder_col_names[field]:
                col_subset.append(field)

        if self.workorder_col_names['additionalCols'] is not None:
            self.mntn_data = self.mntn_data_all[col_subset + self.workorder_col_names['additionalCols']]
        else:
            self.mntn_data = self.mntn_data_all[col_subset]

        # Enforce data type on datetime columns
        self.mntn_data[['CREATION_DATE', 
                        'BASIC_START_DATE',
                        'BASIC_FINISH_DATE',
                        'ACTUAL_START_DATE',
                        'ACTUAL_FINISH_DATE']] = self.mntn_data[['CREATION_DATE',
                                                                'BASIC_START_DATE',
                                                                'BASIC_FINISH_DATE',
                                                                'ACTUAL_START_DATE',
                                                                'ACTUAL_FINISH_DATE']].astype('datetime64[ns]')

        # Remove non-alphanumerical characters (except for select few)
        self.mntn_data['WO_DESCRIPTION'] = self.mntn_data['WO_DESCRIPTION'].apply(lambda x: re.sub(r'[^0-9A-Za-z-*\/ ]', ' ', str(x)))
        # Remove multiple whitespaces
        self.mntn_data['WO_DESCRIPTION'] = self.mntn_data['WO_DESCRIPTION'].apply(lambda x: ' '.join(x.split()))
        # Lower case work order descriptions to match tokens in gazetteer
        self.mntn_data['WO_DESCRIPTION'] = self.mntn_data['WO_DESCRIPTION'].apply(lambda x: str(x).lower())
        
        # Get mean tokens per doc
        doc_lens = [len(x.split()) for x in self.mntn_data['WO_DESCRIPTION'].tolist()]

        print(f'{datetime.now()}: Mean token length - {sum(doc_lens)/len(doc_lens)}')

        # Drop NA values
        if self.filters['dropNA']:
            # WO Description
            self.mntn_data.dropna(subset=['WO_DESCRIPTION'], how='all', inplace=True)
            # Actual Start Time
            self.mntn_data.dropna(subset=['ACTUAL_START_DATE'], how='all', inplace=True)
            # Cost, Time or Cost AND Time
            if set(['TOTAL_ACTUAL_HOURS', 'TOTAL_ACTUAL_COST']).issubset(set(self.mntn_data.columns)):
                print(f'{datetime.now()}: Dataset has both cost and time')
                self.mntn_data.dropna(subset=['TOTAL_ACTUAL_HOURS', 'TOTAL_ACTUAL_COST'], how='all', inplace=True)
            elif 'TOTAL_ACTUAL_HOURS' in self.mntn_data.columns:
                print(f'{datetime.now()}: Dataset has only time')
                self.mntn_data.dropna(subset=['TOTAL_ACTUAL_HOURS'], how='all', inplace=True)
            elif 'TOTAL_ACTUAL_COST' in self.mntn_data.columns:
                print(f'{datetime.now()}: Dataset has only cost')
                self.mntn_data.dropna(subset=['TOTAL_ACTUAL_COST'], how='all', inplace=True)
            else:
                # No structured data
                pass
            
        # Filter based on creation date (in particular for CMMS transitions which aggregate historical WOs on a point transition time)
        if self.filters['dateLower'] is not None:
            self.mntn_data = self.mntn_data[(self.filters['dateLower'] <= self.mntn_data['CREATION_DATE'])]

        # Filter based on particular asset/object type
        if self.filters['objectType']:
            print(f'{datetime.now()}: Filtering dataset for object/asset types - {self.filters["objectType"]}')
            self.mntn_data = self.mntn_data[self.mntn_data['OBJECT_TYPE'].apply(lambda x: str(x).lower()).str.contains(self.filters['objectType'])]

        print(f'{datetime.now()}: Original Size: {start_size} -> Filtered Size: {len(self.mntn_data)}')

    def return_to_function(self):
        """ Determines return to function (RTF) of workorder 
        
            State changing activities ONLY. Material changes (condition -> condition)

            We are not accounting for repair activities

            Future validation methods:
                # Proactive -> IF NOT STRUCTURED FORMAT -> REGEX
                # Reactive -> IF TERM IN TERMS LIST RELATES TO REBUILD/REPAIR/REPLACE/OVERHAUL
        """

        if self.cmms_type == 'JDE':
            # For JDE CMMS there is sufficient information available to determine RTF classifications
            # NLP will be used for the validation process (TODO: Future work)
            ACTIVITY_TYPE_LIST = ['REPAIR', 'OVERHAUL', 'REBUILD']  # Note: JDE doesn't have REPLACE; This is JDE specific, not an available field in SAP/1SAP
            
            # Filter out repair events. TODO: Make robust for cost OR time
            if 'TOTAL_ACTUAL_HOURS' in self.mntn_data.columns:
                self.mntn_data = self.mntn_data.drop(self.mntn_data[(self.mntn_data['ACTIVITY_TYPE'] == 'REPAIR') & (self.mntn_data['TOTAL_ACTUAL_HOURS'] < self.filters['hourMin'])].index)
            else:
                self.mntn_data = self.mntn_data.drop(self.mntn_data[(self.mntn_data['ACTIVITY_TYPE'] == 'REPAIR')].index)
                
            self.mntn_data['RETURN_TO_FUNCTION'] = np.where(((self.mntn_data['ACTIVITY_CAUSE'] == 'PREVENTIVE') & (self.mntn_data['ACTIVITY_TYPE'].isin(ACTIVITY_TYPE_LIST))),
                                                            'PROACTIVE',
                                                            np.where(((self.mntn_data['ACTIVITY_CAUSE'] == 'CORRECTIVE') & (self.mntn_data['ACTIVITY_TYPE'].isin(ACTIVITY_TYPE_LIST))),
                                                            'REACTIVE',
                                                            np.nan))

        if self.cmms_type in ['SAP', '1SAP']:
            # For SAP CMMS where RTF classifications cannot be determined as sufficient detail
            # is not available, NLP will be used to identify Repair/Replace/Overhaul/Rebuil activity
            # types from work order text.

            # column RETURN_TO_FUNCTION_SUPERSET
            # change names
            #     PM01/PM03 -> REACTIVE_SUPERSET -> RTF ONLY USING NLP
            #     PM02 -> PROACTIVE_SUPERSET -> RTF ONLY USING NLP

            # TODO: Implement the NLP validation as specified above. In the meantime, using PM01/02/03 as proxies for state changing activities...

            # Filter out work orders with cost under threshold. TODO: Make robust for cost OR time
            if 'TOTAL_ACTUAL_COST' in self.mntn_data.columns:
                self.mntn_data = self.mntn_data.drop(self.mntn_data[(self.mntn_data['TOTAL_ACTUAL_COST'] < self.filters['costMin'])].index)
            else:
                pass

            # Assuming that PM01/PM02/PM03 are perfect indicators of state changing events...
            self.mntn_data['RETURN_TO_FUNCTION'] = np.where((self.mntn_data['WO_CLASSIFICATION'] == 'PM02'),
                                                            'PROACTIVE',
                                                            np.where((self.mntn_data['WO_CLASSIFICATION'] == 'PM01') | (self.mntn_data['WO_CLASSIFICATION'] == 'PM03'), 
                                                            'REACTIVE',
                                                            np.nan))

    def token_match(self):
        """ Matches _Repair and _Replace tokens in gazetteers with work order free-text """
        self.mntn_data['REPAIR_OR_REPLACE'] = np.where(self.mntn_data['WO_DESCRIPTION'].apply(lambda x: any(item for item in self.repair_selection if item in x)),
                                                        'REPAIR', 
                                                        np.where(self.mntn_data['WO_DESCRIPTION'].apply(lambda x: any(item for item in self.replace_selection if item in x)),
                                                        'REPLACE',
                                                        np.nan))

    def identify_fs(self):
        """ Identifies failures and suspensions in work order data and calculated columns """
        # TODO: Repair doesn't matter in EOL terms as we do not use them in either case
        try:
            self.mntn_data['FAILURE_OR_SUSPENSION'] = np.where((self.mntn_data['RETURN_TO_FUNCTION'] == 'REACTIVE') & (self.mntn_data['REPAIR_OR_REPLACE'] == 'REPLACE'),
                                                                'FAILURE',
                                                                np.where((self.mntn_data['RETURN_TO_FUNCTION'] == 'PROACTIVE') & (self.mntn_data['REPAIR_OR_REPLACE'] == 'REPLACE'), 
                                                                'SUSPENSION',
                                                                np.nan))
        except Exception as e:
            print(f'{datetime.now()}: Error identifying failure and suspensions - {e}')


class ParameterEstimator(DataWrangler):
    """ Outputs MTBF, Beta, Eta and associated failure instances (free-text) in JSON"""
    def __init__(self, config):
        DataWrangler.__init__(self, config)

        self.config = config
        self.output_prefix = config['Data']['outputPrefix']
        self.no_data_points = config['Calculation']['noDataPoints']

        self.preprocess_fs_data()
        self.calculate_mtbf()
        self.save_results()

    def preprocess_fs_data(self):
        """ Extracts only relevant information from work order data for reliability parameter estimations """
        
        # Group by asset and time, sort newest to oldet (reference is ACTUAL_START_DATE). TODO: Validate the date is correct (was CREATION_DATE)
        self.mntn_data = self.mntn_data.sort_values(by=['FUNCTIONAL_LOC', 'ACTUAL_START_DATE'], ascending=True)

        # Create a mask for only F or S data (exlcudes NaN)
        fs_mask = ((self.mntn_data['FAILURE_OR_SUSPENSION'] == 'FAILURE') | (self.mntn_data['FAILURE_OR_SUSPENSION'] == 'SUSPENSION'))

        # Subset original dataset with mask
        self.mntn_data_fs = self.mntn_data[fs_mask]

        # Calculate the time between failures and suspensions
        # This is done via groupby shift to get date between FS events
        self.mntn_data_fs['TIME'] = self.mntn_data_fs.groupby('FUNCTIONAL_LOC')['ACTUAL_START_DATE'].diff() / np.timedelta64(1, 'D')
        self.mntn_data_fs['TIME'] = self.mntn_data_fs['TIME'].fillna(0)

        # Add Time data to main dataframe for output later on
        self.mntn_data = pd.concat([self.mntn_data, self.mntn_data_fs['TIME']], axis=1)
        # Round Time
        self.mntn_data['TIME'] = self.mntn_data['TIME'].apply(lambda x: x if math.isnan(x) else int(x))

        # Subset dataframe keeping only columns necessary for Weibull calculations
        self.mntn_data_fs_ss = self.mntn_data_fs[['FUNCTIONAL_LOC', 'FAILURE_OR_SUSPENSION', 'TIME']]

        def count_failure(series):
            return series[series == 'FAILURE'].count()
        def count_suspension(series):
            return series[series == 'SUSPENSION'].count()

        # Aggregate - groupby asset_number and count failure/suspensions to add for meta data in the excel output
        # This is used for validation of F and S counts
        self.mntn_data_fs_ss_counts = self.mntn_data_fs_ss.groupby(['FUNCTIONAL_LOC'])['FAILURE_OR_SUSPENSION'].agg(['count', count_failure, count_suspension])

    def calculate_mtbf(self):
        """ Calculates MTBF from two parameter Weibull 

            This is adapted from the following notebook: https://github.com/uwasystemhealth/weibull-python/blob/master/weibull-python.ipynb
            
            Note: Failure (F) = 1 and Suspension (S) = 0
        """

        fs_data = self.mntn_data_fs_ss.copy()

        # Adjusting column names and binarising F/S strings
        fs_data['FAILURE_OR_SUSPENSION'].replace(to_replace='FAILURE', value=1, inplace=True)
        fs_data['FAILURE_OR_SUSPENSION'].replace(to_replace='SUSPENSION', value=0, inplace=True)

        # Only objects/assets with a combined F/S count under that specified in config will be considered herein.
        fs_data_ss = fs_data.copy()

        # Remove any groupby object with less than specified points
        fs_data_ss = fs_data_ss.groupby(['FUNCTIONAL_LOC']).filter(lambda x: self.no_data_points <= len(x))

        def compute_single_mtbf(series):
            """
            Computes a single MTBF estimate via 2P Weibull plot fitting.
            
            Notes:
            - All events with 0.0 time (first occurences) are filtered out
            - How do we deal with only censored events?
            - For censored and failure data, does the Fit_Weibull_2P AND wb.Analysis need both times? see note below.
            
            Arguments
            ---------
            series : Pandas Series object
                Event occurence and time
            Returns
            -------
            MTBF : float
                mean time to failure
            """
            series = series[0 < series['TIME']]   # do not take into account any events that have 0 time...
            
            failures = series[series['FAILURE_OR_SUSPENSION']==1]
            right_censored = series[series['FAILURE_OR_SUSPENSION']==0]
            
            failure_times = failures['TIME'].values.tolist()
            right_censored_times = right_censored['TIME'].values.tolist()
            
            # Fit Weibull
            if (1 < len(failures)) & (1 < len(right_censored)): 
                wbfit = rb.Fitters.Fit_Weibull_2P(failures=failure_times,
                                                right_censored=right_censored_times,
                                                show_probability_plot=False)
                # should this analysis below have censored data in it? it isnt in the original notebook
                analysis = wb.Analysis(data=failure_times)
                analysis.fit()
                analysis.beta = wbfit.beta
                analysis.eta = wbfit.alpha
                return pd.Series([analysis.mtbf, analysis.beta, analysis.eta])
                
            elif (1 < len(failures)):
                wbfit = rb.Fitters.Fit_Weibull_2P(failures=failure_times, show_probability_plot=False)
                analysis = wb.Analysis(data=failure_times)
                analysis.fit()
                analysis.beta = wbfit.beta
                analysis.eta = wbfit.alpha
                return pd.Series([analysis.mtbf, analysis.beta, analysis.eta])
            else:
                # print('ONLY CENSORED EVENTS')
                pass
        
        # Init empty dataframe
        self.mtbf_data = pd.DataFrame()

        # Perform MTBF calculations on all asset groups
        self.mtbf_data[['MTBF', 'BETA', 'ETA']] = fs_data_ss.groupby(['FUNCTIONAL_LOC']).apply(lambda grp: compute_single_mtbf(grp))
        
    def save_results(self):
        """ Convert data to JSON and save to disk """

        total_cost_or_time_col = 'TOTAL_ACTUAL_COST' if 'TOTAL_ACTUAL_COST' in self.mntn_data.columns else 'TOTAL_ACTUAL_HOURS'
        
        # If no cost or time information is specified, set as 0.
        if total_cost_or_time_col not in self.mntn_data.columns:
            self.mntn_data[total_cost_or_time_col] = 0

        # Init results dictionary
        output = dict()

        # Add calculation parameters to results
        output['SETTINGS'] = {
            'Object Type': self.filters['objectType'],
            'Thresholds': {'Cost': self.filters['costMin'],
                            'Time': self.filters['hourMin'],
                            'Points': self.no_data_points,
                            'Lower Date': self.filters['dateLower'],
                            'Upper Date': self.filters['dateUpper']
                            }
                        }

        # High level analysis overview
        output['OVERVIEW'] = {
            'TOTAL_WORKORDERS': len(self.mntn_data),
            'FAILURES': int(self.mntn_data_fs_ss_counts['count_failure'].sum()),
            'SUSPENSIONS': int(self.mntn_data_fs_ss_counts['count_suspension'].sum()),
            'TOTAL_ASSETS': self.mntn_data['FUNCTIONAL_LOC'].nunique(),     # number of assets that went through the filters etc.
            'MTBF_ASSETS': self.mtbf_data.index.nunique()                 # number of assets that had MTBFs calculated
        }
        
        # FLOC specific results and data
        output['RESULTS'] = dict()
        for floc in self.mtbf_data.index:
            output['RESULTS'][floc] = {
                'MTBF': round(self.mtbf_data.loc[floc]['MTBF'],2),
                'ETA': round(self.mtbf_data.loc[floc]['ETA'],2),
                'BETA': round(self.mtbf_data.loc[floc]['BETA'],2),
                'COUNTS': {
                    'FAILURE': int(self.mntn_data_fs_ss_counts.loc[floc]['count_failure']),
                    'SUSPENSION': int(self.mntn_data_fs_ss_counts.loc[floc]['count_suspension']),
                    'TOTAL': len(self.mntn_data[self.mntn_data['FUNCTIONAL_LOC'] == floc])
                },
                # TODO: Fix issue with having to filter nan via 'nan' strings rather than .notnull() or .isnan()...
                'WO_DESCRIPTIONS_FS': self.mntn_data[(self.mntn_data['FUNCTIONAL_LOC'] == floc) & (self.mntn_data['FAILURE_OR_SUSPENSION'] != 'nan')][['WO_DESCRIPTION', total_cost_or_time_col, 'WO_CLASSIFICATION', 'RETURN_TO_FUNCTION', 'REPAIR_OR_REPLACE','FAILURE_OR_SUSPENSION', 'TIME']].to_dict(),
                'WO_DESCRIPTION_NFS': self.mntn_data[(self.mntn_data['FUNCTIONAL_LOC'] == floc) & (self.mntn_data['FAILURE_OR_SUSPENSION'] == 'nan')][['WO_DESCRIPTION', total_cost_or_time_col, 'WO_CLASSIFICATION', 'RETURN_TO_FUNCTION', 'REPAIR_OR_REPLACE','FAILURE_OR_SUSPENSION', 'TIME']].to_dict() # Filtered using NaN in F/S column
            }

        with open(Path(self.config['File']['outputDir']) / 'mtbf_results.json', 'w') as fp:
            json.dump(output, fp, indent = 4)
            
        if self.config['File']['saveCSV']:
            cost_or_hours = 'hours' if self.config['Data'].get('TOTAL_ACTUAL_HOURS') else 'cost'
            json2csv(Path(self.config['File']['outputDir']) / 'mtbf_results.json', Path(self.config['File']['outputDir']) / 'mtbf_results.csv', cost_or_hours)
        
            

def controller(config_path: str):
    start_time = time.time()
    config = load_config(path = config_path)
    ParameterEstimator(config)
    print(f'{datetime.now()}: Process took - {(time.time() - start_time):0.2f}s to complete.')

if __name__ == '__main__':
    config_path = 'config_template.yml'
    controller(config_path)