# python pre_run.py tokens
# python main_bm25.py

from utils.main import *
from rank_bm25 import BM25Okapi, BM25Plus
from nltk.corpus import wordnet
from itertools import chain

def expand_query_with_synonyms(query_tokens):
    expanded_query = list(query_tokens)
    for token in query_tokens:
        synonyms = wordnet.synsets(token)
        lemmas = list(chain.from_iterable([word.lemma_names() for word in synonyms]))
        expanded_query.extend(lemmas)
    return expanded_query

def bm25_proccess(query_tokens, useful_preprocessed_files, k1=2, b=0.66):
    bm25 = BM25Plus(useful_preprocessed_files.values(), k1=k1, b=b) # 0.3988
    
    docnos = useful_preprocessed_files.keys()
    scores = bm25.get_scores(query_tokens)
    sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True)
    return sorted_doc_scores
        
def compute_scores_for_query(args):
    num, query_tokens, preprocessed_files_tokens, inverted_index, K1, B = args
    useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='tokens')
    sorted_doc_scores = bm25_proccess(query_tokens, useful_preprocessed_files, k1=K1, b=B)
    return num, sorted_doc_scores

def compute_scores_for_chunk(chunk):
    results_for_chunk = [compute_scores_for_query(args) for args in chunk]
    return results_for_chunk
    
def main():
    preprocessed_files_tokens = load_from_json("saved_preprocessed_files_tokens.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries', type='tokens')
    
    all_doc_scores = {}
    K1, B = 2, 0.66 # 1.2 ≤ k1 ≤ 2.0 and 0.5 ≤ b ≤ 0.75
    
    all_data = [(num, query_tokens, preprocessed_files_tokens, inverted_index, K1, B) for num, query_tokens in preprocessed_queries.items()]
    chunk_size = 6
    query_chunks = list(chunks(all_data, chunk_size))
    with ProcessPoolExecutor() as executor:
        for chunk_result in executor.map(compute_scores_for_chunk, query_chunks):
            for num, sorted_doc_scores in chunk_result:
                all_doc_scores[int(num)] = sorted_doc_scores
    
    sorted_all_doc_scores = sorted(all_doc_scores.items(), key=lambda x: x[0])
    for num, sorted_doc_scores in sorted_all_doc_scores:
        write_results_into_text_file(num, sorted_doc_scores, run_name='bm25')
        
    print(f'Running trec_eval for file result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
                
if __name__ == "__main__":
    main()
