'''
Master controller for pipeline

@author: Tyler Bikaun
'''

import term_expansion
import collocations
import mwo_to_mtbf

def main(config_path):
    term_expansion.controller(config_path)  # uses w2v and SME to output token_file.xlsx
    collocations.controller(config_path)    # Adds collocations to token file
    mwo_to_mtbf.controller(config_path)     # performs parameter estimation over MWOs

if __name__ == '__main__':
    config_path = 'config_template.yml'
    main(config_path)