"""
Expands initial terms list and validates using contextual word embedding techniques

@author: Tyler Bikaun
"""
# Standard libraries
import yaml
import pandas as pd
import numpy as np
import sys, traceback
import time
import re
from datetime import datetime
from pathlib import Path

# NLP specific libraries
import nltk
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from utils import load_config

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.config['File']['workorderPath'])
        df = df[self.config['Data']['workorderColNames']['WO_DESCRIPTION']].str.lower()
        shorttext_docs = df.tolist()
        
        # Remove non-alphanumerical characters (except for select few)
        shorttext_docs = [re.sub(r'[^0-9A-Za-z-*\/ ]', ' ', str(doc)) for doc in shorttext_docs]
        # Remove multiple whitespace
        shorttext_docs = [' '.join(doc.split()) for doc in shorttext_docs]
        # Tokenize data on single whitespace
        self.data = [doc.split(' ') for doc in shorttext_docs]
    
        print(f'Processed {len(self.data)} MWOs')

    def get_stats(self):
        doc_lens = [len(doc) for doc in self.data]
        print(f'Average Tokens in MWOs: {sum(doc_lens)/len(doc_lens):0.2f} ({max(doc_lens)} max | {min(doc_lens)} min)')


class EmbeddingTrainer:
    def __init__(self):
        self.min_token_count = 2
        self.window_size = 3
        self.model_size = 300
    
    def train_model(self, docs, iterations: int = 50):
        ''' '''
        w2v_model = Word2Vec(sentences = docs,
                                size = self.model_size,
                                min_count = self.min_token_count,
                                window = self.window_size,
                                iter = iterations
                                )
        print(f'Model summary:\n {w2v_model}')
        return w2v_model
            
class EOLExpander:
    def __init__(self):
        pass
    
    def expand_terms(self, model, docs, terms, topn: int = 10, mode: str = '_Replace'):
        ''' Uses word embeddings to expand set of terms within documents '''
        
        # for term in initial_terms
        #   get similar terms to term
        #   filter for new terms (not in initial_terms)
        #   select additional terms
        #   capture additional terms
        #   capture irrelevant terms diff(set(add_terms), set(similar terms))
        
        active = True
        start_time = time.time()
        print(f'{datetime.now()}: Expanding terms in {mode}')
        
        while active:
            try:
                def get_similar_terms(term, irrelevant_terms, topn):
                    ''' Iteratively searches for topn similar terms whilst negating irrelevant terms '''
                    
                    tries = 0
                    max_tries = 100
                    similar_terms = []
                    similar_terms_proba = []
                    while len(similar_terms) <= topn:
                        similar_terms_temp, similar_terms_proba_temp = zip(*model.most_similar([term], topn = topn*10))#(topn-len(similar_terms))))
                        
                        # remove similar terms already in terms list
                        new_terms_idx = [idx for idx, similar_term in enumerate(similar_terms_temp) if similar_term not in irrelevant_terms]
                        # slice terms and proba lists
                        similar_terms_temp_slice = np.array(list(similar_terms_temp))[new_terms_idx]
                        similar_terms_proba_temp_slice = np.array(list(similar_terms_proba_temp))[new_terms_idx]

                        similar_terms.extend(similar_terms_temp_slice)
                        similar_terms_proba.extend(similar_terms_proba_temp_slice)
                    
                        tries += 1
                        
                        if tries == max_tries:
                            return None, None
                    
                    return similar_terms[:topn], similar_terms_proba[:topn]
                
                # Lower case term
                terms = [term.lower() for term in terms]
                
                irrelevant_terms = []
                additional_terms = []
                for term in terms:
                    try:
                        similar_terms, similar_terms_proba = get_similar_terms(term, irrelevant_terms=terms+irrelevant_terms, topn=topn)
                        
                        if similar_terms is None:
                            print(f'No term in embedding model for {term}')
                        else:
                            print('\n', '*'*100)
                            print('Terms matching:', term)
                            print('*'*100)
                            
                            print("\t".join([f'{similar_terms[idx]} ({similar_terms_proba[idx]*100:0.0f}%)' for idx, _ in enumerate(similar_terms)]))
                            
                            new_terms = input('Enter terms to add to list (each term separated by a space):\n')
                            if new_terms != '':
                                if len(new_terms.split()) > 1:
                                    print('Multiple terms')
                                    for idx, _ in enumerate(new_terms.split()):
                                        additional_terms.append(new_terms.split()[idx])
                                    # Add additional terms to terms list
                                    terms.extend(new_terms.split())
                                else:
                                    print('Single term')
                                    if len(new_terms) > 1:
                                        additional_terms.append(new_terms)
                                    # Add additional term to terms list
                                    terms.append(new_terms)
                                
                                # Get irrelevant terms
                                irrelevant_terms.extend(list(set(similar_terms) - set(new_terms.split())))
                            else:
                                irrelevant_terms.extend(list(set(similar_terms)))
                            
                            print(f'Number of irrelevant terms captured: {len(irrelevant_terms)}')
                        
                    except:
                        traceback.print_exc(file=sys.stdout)
                        print(f'No term in embedding model for {term}')


                terms_printout = "\n".join([f'\t- {term}' for term in additional_terms])
                print('*'*20)
                print(f'Added {len(additional_terms)} terms to gazetteer:\n{terms_printout}')
                
                terms.extend(list(set(additional_terms)))
                terms_out = "\n".join(list(set(terms)))
                active = False  # Finished going through the terms

            except KeyboardInterrupt:
                active = False
            
        print(f'Captured {len(additional_terms)} additional terms in {time.time() - start_time:0.0f}s')
        
        # Save terms
        df = pd.DataFrame(data={'Term': terms, 'Action': [mode]*len(terms)})
        
        return df

def controller(config_path):
    config = load_config(path = config_path)
    # Instatiate classes
    dl = DataLoader(config)
    trainer = EmbeddingTrainer()
    expander = EOLExpander()
    
    # Get embedding model
    w2v = trainer.train_model(docs = dl.data, iterations = 100)

    # Get new terms
    initial_terms_replace = input('Enter initial replace terms (each term separated by a space):\n')
    terms_list_replace = initial_terms_replace.split()
    df_replace = expander.expand_terms(model = w2v, docs = dl.data, terms = terms_list_replace, mode = '_Replace')

    initial_terms_repair = input('\nEnter initial repair terms (each term separated by a space):\n')
    terms_list_repair = initial_terms_repair.split()

    df_repair = expander.expand_terms(model = w2v, docs = dl.data, terms = terms_list_repair, mode = '_Repair')

    df = pd.concat([df_replace, df_repair])
    df.drop_duplicates(inplace = True)
    
    # Save token list to disk and return as df to downstream components
    token_path = Path(config['File']['outputDir']) / 'token_file.xlsx'
    df.to_excel(token_path, index = False)


if __name__ == '__main__':
    config_path = './exp/A/config_A_S2.yml'
    controller(config_path)