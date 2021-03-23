'''
Script for identifying bigram collocations.

@author: Tyler Bikaun
'''

import yaml
import itertools
from collections import Counter
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pandas as pd

from term_expansion import DataLoader
from utils import load_config

def get_collocations(docs: list, n: int = 100):
    ''' Find top collocations in text. These are then converted into conjunctions with a delimiter.
    "Collocations are expressions of multiple words which commonly co-occur."
    
    Note: counters can have mathematical operations applied to them.
    
    
    Ref: https://stackoverflow.com/questions/4128583/how-to-find-collocations-in-text-python
    '''
    
    active = True
    collocations_kept = []

    collocations = Counter()
    for words in tqdm(docs):
        nextword = iter(words)
        next(nextword)
        freq = Counter(zip(words, nextword))
        collocations += freq
    
    # Show user top-n collocations
    while active:
        for col in collocations.most_common(n):
            col_ngram = f'{col[0][0]} {col[0][1]}'
            
            output_str =f'{col_ngram:<20} : {col[1]:<5} ({(int(col[1])/len(docs))*100:0.2f}%)'

            decision = input(f'{output_str} | Add to collocations (Y/N)? ')
            decision = decision.lower()
            
            if decision == 'y':
                collocations_kept.extend([col_ngram])
            
        active = False
        
    return collocations_kept

def controller(config_path: str):
    config = load_config(config_path)
    # load data
    dl = DataLoader(config)
    
    # Get collocations
    collocations = get_collocations(docs = dl.data, n = config['Settings']['collocationCandidates'])
    
    # Add to token file
    token_df = pd.read_excel(Path(config['File']['outputDir']) / 'token_file.xlsx')
    
    if len(collocations) > 0:
        collocation_df = pd.DataFrame(data=zip(collocations, ['_Replace']*len(collocations)), columns=['Term', 'Action'])
        # Add collocations to tokens
        token_df = pd.concat([token_df, collocation_df], ignore_index=True)
        # Save token list to disk and return as df to downstream components
        token_path = Path(config['File']['outputDir']) / 'token_file.xlsx'
        token_df.to_excel(token_path, index = False)
        
    print(f'{datetime.now()}: {len(collocations)} collocations added to token file')

if __name__ == '__main__':
    config_path = 'config_template.yml'
    controller(config_path)