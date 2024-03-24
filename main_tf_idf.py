# python pre_run.py tokens
# python main_tf_idf.py

from utils.main import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity

def tf_idf_proccess(query_tokens, useful_preprocessed_files):
    dictionary = Dictionary(useful_preprocessed_files.values())
    corpus = [dictionary.doc2bow(doc) for doc in useful_preprocessed_files.values()]
    
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    docnos = useful_preprocessed_files.keys()
    
    query_bow = dictionary.doc2bow(query_tokens)
    query_tfidf = tfidf[query_bow]
    
    corpus_matrix = matutils.corpus2csc(corpus_tfidf).T
    query_matrix = matutils.corpus2csc([query_tfidf], num_terms=len(dictionary)).T
    
    scores = cosine_similarity(query_matrix, corpus_matrix)[0]
    
    sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
    return sorted_doc_scores

def main():
    preprocessed_files_tokens = load_from_json("saved_preprocessed_files_tokens.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries', type='tokens')
    
    for num, query_tokens in preprocessed_queries.items():
        useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='tokens')
        sorted_doc_scores = tf_idf_proccess(query_tokens, useful_preprocessed_files)
        write_results_into_text_file(num, sorted_doc_scores, run_name='tf_idf')
        
    print(f'Running trec_eval for result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
    
if __name__ == "__main__":
    main()
