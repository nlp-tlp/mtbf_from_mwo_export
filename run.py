'''
Master controller for pipeline.

@author: Tyler Bikaun
'''

from pipeline import term_expansion, collocations, mwo_to_mtbf

def main(config_path: str):
    ''' 
    Executes pipeline for MTBF estimation from maintenance records.
    
    Parameters
    ----------
    config_path: Str
        Path to configuration file

    '''
	
    term_expansion.controller(config_path)  # Uses w2v and SME to output token_file.xlsx
    collocations.controller(config_path)    # Adds collocations to token file
    mwo_to_mtbf.controller(config_path)     # Performs parameter estimation over MWOs

if __name__ == '__main__':
    config_path = 'config_template.yml'
    main(config_path)