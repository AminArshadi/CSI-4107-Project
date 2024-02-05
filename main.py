from utils.main import *
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import numpy as np
import subprocess

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
    
    k1_values = np.arange(2, 2.1, 0.1) # k1 between 1 and 2 (1, 2.1, 0.1)
    b_values = np.arange(7.4, 7.6, 0.01) # b between 0.3 and 0.9 (0.3, 1, 0.1)
    
    saved_result, max_map = None, 0.0
    for k1 in k1_values:
        k1 = round(k1, 2)
        result = {}
        
        for b in b_values:
            b = round(b, 2)
            
            bm25 = BM25Okapi(useful_preprocessed_files.values(), k1=k1, b=b)
            # bm25 = BM25L(useful_preprocessed_files.values(), k1=k1, b=b)
            # bm25 = BM25Plus(useful_preprocessed_files.values(), k1=k1, b=b)

            print('Making the Results file ...')
            result_file = open(f'./trec_eval/results/results_{k1}_{b}.txt', 'w')
            
            for num, query_tokens in preprocessed_queries.items():
                docnos = preprocessed_files.keys()
                scores = bm25.get_scores(query_tokens)
                sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
                
                for rank, (docno, score) in enumerate(sorted_doc_scores):
                    result_file.write(f'{num} Q0 {docno} {rank + 1} {score} {k1}_{b}\n')
            
            result_file.close()
            print('Done.')
            
            print(f'Running trec_eval for file results_{k1}_{b}.txt ...')
            executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', f'./trec_eval/results/results_{k1}_{b}.txt']
            process = subprocess.Popen(executable, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout.decode()
            
            for elem in output.split('\n'):
                elem = elem.split('\t')
                if len(elem) > 1:
                    result[elem[0].strip()] = elem[-1].strip()
            
            new_map = float(result['map'])
            if new_map > max_map:
                max_map = new_map
                saved_result = result
                
            print(f'result_map = {new_map}')
            print(f'max_map = {max_map}')
                
            if stderr:
                print("Errors:", stderr.decode())
            print('Done.')
    
    print('#################')
    print('Highest MAP Score:')
    pprint(saved_result)
    print('#################')
    
if __name__ == "__main__":
    main()
