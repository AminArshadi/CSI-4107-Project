from utils.main import *
from rank_bm25 import BM25Okapi

def main():
    preprocessed_files = load_from_json("saved_preprocessed_files.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries')
    useful_docnos = []
    
    for _, query_tokens in preprocessed_queries.items():
        for query_token in query_tokens:
            docnos = inverted_index.get(query_token, None)
            if docnos is not None:
                useful_docnos.extend(docnos)
            
    useful_preprocessed_files = {useful_docno: preprocessed_files[useful_docno] for useful_docno in list(set(useful_docnos))}
    
    bm25 = BM25Okapi(useful_preprocessed_files.values())
    
    print('Making the Results file ...')
    result_file = open('Results.txt', 'w')
    
    for num, query_tokens in preprocessed_queries.items():
        docnos = preprocessed_files.keys()
        scores = bm25.get_scores(query_tokens)
        sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
        
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            result_file.write(f'{num} Q0 {docno} {rank + 1} {score} run_name_{rank}\n')
    
    result_file.close()
    print('Done.')
    
if __name__ == "__main__":
    main()
