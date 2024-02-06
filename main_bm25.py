from utils.main import *
from rank_bm25 import BM25Okapi, BM25Plus

def bm25_proccess(num, query_tokens, useful_preprocessed_files, k1=1.5, b=0.75):
    # bm25 = BM25Okapi(useful_preprocessed_files.values()) # 0.2960
    bm25 = BM25Plus(useful_preprocessed_files.values()) # 0.2987

    docnos = useful_preprocessed_files.keys()
    scores = bm25.get_scores(query_tokens)
    sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
    
    print(f'Adding the results of query {num} to the results_{k1}_{b}.txt file ...')
    with open(f'./trec_eval/results/results_{k1}_{b}.txt', 'a') as result_file:
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            result_file.write(f'{num} Q0 {docno} {rank + 1} {score} {k1}_{b}\n')
    print('Done.')
    
    return
    
def main():
    preprocessed_files = load_from_json("saved_preprocessed_files.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries')
    
    k1, b = 1.5, 0.75
    
    for num, query_tokens in preprocessed_queries.items():
        useful_docnos = []
        for query_token in query_tokens:
            docnos = inverted_index.get(query_token, None)
            if docnos is not None:
                useful_docnos.extend(docnos)
            
        useful_preprocessed_files = {useful_docno: preprocessed_files[useful_docno] for useful_docno in list(set(useful_docnos))}
        bm25_proccess(num, query_tokens, useful_preprocessed_files, k1=k1, b=b)
        
    print(f'Running trec_eval for file results_{k1}_{b}.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', f'./trec_eval/results/results_{k1}_{b}.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
                
if __name__ == "__main__":
    main()
