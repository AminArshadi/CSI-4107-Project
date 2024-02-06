from utils.main import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity

def tf_idf_proccess(num, query_tokens, useful_preprocessed_files):
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
    
    print(f'Adding the results of query {num} to the result.txt file ...')
    with open('./trec_eval/results/result.txt', 'a') as result_file:
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            result_file.write(f'{num} Q0 {docno} {rank + 1} {score} run_name\n')
    print('Done.')
    
    return

def main():
    preprocessed_files = load_from_json("saved_preprocessed_files.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries')
    
    for num, query_tokens in preprocessed_queries.items():
        useful_docnos = []
        for query_token in query_tokens:
            docnos = inverted_index.get(query_token, None)
            if docnos is not None:
                useful_docnos.extend(docnos)
            
        useful_preprocessed_files = {useful_docno: preprocessed_files[useful_docno] for useful_docno in list(set(useful_docnos))}
        tf_idf_proccess(num, query_tokens, useful_preprocessed_files)
        
    print(f'Running trec_eval for result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', f'./trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
    
if __name__ == "__main__":
    main()
