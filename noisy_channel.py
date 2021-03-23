"""
Neural noisy channel model development and training for denoising documents

@author: Tyler Bikaun
"""
import math

terms_list = set(['c/ out','c/o','change','chg','chge','fabr','fit','install','o/ haul','o/h','over haul','overhaul','ovhl','r & r','R&R','recover','remove','repl','replace','seized','siezed','swap','u/s','remove/replace','reinstall','breakdown-change','c/out','changeout','change-out','c/out.','u/s.','replace.','change','c-out','repair/replace','/replace','o/haul','overhaul.','seized.','seized,','changing','snapped','blown','fit_new','install_new','change_out','change_out.','needs_replacing','over_haul','failed.','*breakdown*','us','failed'])

data_path = 'data/bhp/odr-1sap-work-management-work-order-21.10.2020-pumps.txt'

with open(data_path) as f:
    mwo_docs = list()
    for line in f.readlines():
        mwo_docs.append(line.lower())

def get_term_freq(docs):
    term_freq = dict()
    for doc in docs:
        tokens = doc.split()
        for token in tokens:
            if term_freq.get(token):
                term_freq[token] += 1
            else:
                term_freq[token] = 1

    return term_freq

def get_vocab(term_freq):
    return set(list(term_freq.keys()))

term_freq = get_term_freq(mwo_docs)

# print(f'Term Frequencies:\n{term_freq}')

print(f'Vocab:\n{get_vocab(term_freq)}')

# Get mean frequency of all tokens
mean_freq = math.ceil(sum(list(term_freq.values()))/len(list(term_freq.values())))

count = 1
docs_wo_noise = 0
all_noised_tokens = list()
for doc in mwo_docs:
    # Convert doc to set based on tokens
    doc_set = set(doc.split())

    # Potential noised tokens
    potential_noised_tokens = list()
    for token in doc_set - terms_list:
        if term_freq.get(token) < mean_freq:
            potential_noised_tokens.append(token)
            all_noised_tokens.append(token)

    if len(potential_noised_tokens) == 0:
        docs_wo_noise += 1

    # Check if any term is in doc via intersection
    matched_terms = terms_list & doc_set
    print(f'\nDoc {count}: {doc} | Matched EOL Token(s): {list(matched_terms)[0] if len(matched_terms) > 0 else "No Matches"} | Potential Noised Tokens: {"/".join(potential_noised_tokens) if len(potential_noised_tokens) > 0 else "No Matches"}')


    count += 1

print(f'Total docs: {count} | Docs without noise: {docs_wo_noise}')
print(f'Word mean frequency cutoff: {mean_freq}')

print("\n".join(all_noised_tokens))
print(f'Unique noised tokens: {len(all_noised_tokens)}/{len(term_freq.values())}')



if __name__ == '__main__':

    pass
