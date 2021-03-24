'''
End-of-life iterative term expansion process using domain-specific pre-trained word embeddings.

@author: Tyler Bikaun
'''


import pandas as pd
import numpy as np
import sys, traceback
import time
import re
from datetime import datetime
from pathlib import Path
from gensim.models import Word2Vec

try:
    from pipeline.utils import load_config
except ImportError:
    from utils import load_config


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.load_csv()

    def load_csv(self):
        ''' 
        Loads data in CSV format from disk and performs light preprocessing and tokenization
        '''
        
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
        ''' 
        Provides average token length of documents in data set
        '''
        
        doc_lens = [len(doc) for doc in self.data]
        print(f'Average Tokens in MWOs: {sum(doc_lens)/len(doc_lens):0.2f} ({max(doc_lens)} max | {min(doc_lens)} min)')


class EmbeddingTrainer:
    def __init__(self):
        self.min_token_count = 2
        self.window_size = 3
        self.model_size = 300
    
    def train_model(self, docs: list, iterations: int = 250):
        ''' 
        Trains for word2vec embedding model and provides summary of results
        
        Parameters
        ----------
        docs : list
            List of tokenized maintenance work order record descriptions
        iterations : int
            Number of iterations to run the word2vec model for
        
        Returns
        -------
        w2v_model : Word2Vec object
        
        Notes
        -----
        See: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

        '''
        w2v_model = Word2Vec(sentences = docs,
                                size = self.model_size,
                                min_count = self.min_token_count,
                                window = self.window_size,
                                iter = iterations
                                )
        print(f'Model summary:\n {w2v_model}')
        return w2v_model


class EOLExpander:
    def expand_terms(self, model, docs: list, terms, topn: int = 10, mode: str = 'replace'):
        ''' 
        Uses word embeddings to expand a set of terms within derived from a set of maintenance work order descriptions docs
        
        Parameters
        ----------
        model : Word2Vec object
            Pre-trained word2vec model
        docs : List
            Set of documents pertaining to maintenance work order descriptions
        terms : TODO: remember what I meant here...
            ...
        topn : Int
            Number of results to provide the user at each iteration
        mode : Str
            Mode for EOL term classification - options: replace or repair
        
        Returns
        -------

        
        Notes
        -----
        Process pseudo-code:
        ```for term in initial_terms
            get similar terms to term
            filter for new terms (not in initial_terms)
            select additional terms
            capture additional terms
            capture irrelevant terms diff(set(add_terms), set(similar terms))
        ```

        '''
        
        active = True
        start_time = time.time()
        print(f'{datetime.now()}: Expanding terms in {mode}')
        
        while active:
            try:
                def get_similar_terms(term: str, irrelevant_terms: list, topn: int):
                    '''
                    Iterative search for topn similar terms whilst negating irrelevant terms 
                    
                    Parameters
                    ----------
                    term : Str
                        Current term used for similarity matching with word2vec model
                    irrelevant_terms : List
                        List of terms deemed irrelevant by user, cached over 
                        the expansion process so they are not presented multiple times
                    topn : Int
                        Number of results to provide the user at each iteration
                    
                    Results
                    -------
                    
                    
                    Notes
                    -----
                    
                    
                    '''
                    
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
                        similar_terms, similar_terms_proba = get_similar_terms(term, irrelevant_terms = terms+irrelevant_terms, topn = topn)
                        
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
                                    # print('Multiple terms')
                                    for idx, _ in enumerate(new_terms.split()):
                                        additional_terms.append(new_terms.split()[idx])
                                    # Add additional terms to terms list
                                    terms.extend(new_terms.split())
                                else:
                                    # print('Single term')
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
                # Finished going through the terms
                active = False

            except KeyboardInterrupt:
                active = False
            
        print(f'Captured {len(additional_terms)} additional terms in {time.time() - start_time:0.0f}s')
        
        # Save terms
        df = pd.DataFrame(data={'Term': terms, 'Action': [mode]*len(terms)})
        
        return df


def controller(config_path: str):
    '''
    Controller for the execution of the entire term expansion process

    Parameters
    ----------
    config_path : Str
        Path to configuration file on disk
    
    '''
    
    config = load_config(path = config_path)
    # Instatiate classes
    dl = DataLoader(config)
    trainer = EmbeddingTrainer()
    expander = EOLExpander()
    
    # Get embedding model
    w2v = trainer.train_model(docs = dl.data, iterations = 100)

    # Get new terms
    initial_termsreplace = input('Enter initial replace terms (each term separated by a space. hint: replace is a good starting word):\n')
    terms_listreplace = initial_termsreplace.split()
    dfreplace = expander.expand_terms(model = w2v, docs = dl.data, terms = terms_listreplace, mode = 'replace')

    # TODO: Determine if keeping this part of the code...
    initial_termsrepair = input('\nEnter initial repair terms (each term separated by a space. hint: repair is a good starting word):\n')
    terms_listrepair = initial_termsrepair.split()
    dfrepair = expander.expand_terms(model = w2v, docs = dl.data, terms = terms_listrepair, mode = 'repair')
    df = pd.concat([dfreplace, dfrepair])
    
    df.drop_duplicates(inplace = True)
    
    # Save token list to disk and return as df to downstream components
    token_path = Path(config['File']['outputDir']) / 'token_file.xlsx'
    df.to_excel(token_path, index = False)

if __name__ == '__main__':
    config_path = 'cofig_template.yml'
    controller(config_path)