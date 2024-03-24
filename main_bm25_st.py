# python pre_run.py tokens
# python main_bm25_st.py

from utils.main import *
from main_bm25 import *
from main_st import *

def get_top_K_docnos(sorted_doc_scores):
    K, top_K_docnos = 1000, []
    
    for rank, (docno, _) in enumerate(sorted_doc_scores, start=1):        
        top_K_docnos.append(docno)
        if rank == K:
            break
    
    return top_K_docnos
    
def main():
    preprocessed_files_tokens = load_from_json("saved_preprocessed_files_tokens.json")
    preprocessed_files_text = load_from_json("saved_preprocessed_files_text.json")
    
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries_tokens = preprocess_queries('./queries', type='tokens')
    preprocessed_queries_text = preprocess_queries('./queries', type='text')
        
    for num, query_tokens in preprocessed_queries_tokens.items():
        useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='tokens')
        sorted_doc_scores = bm25_proccess(num, query_tokens, useful_preprocessed_files)
        
        top_K_docnos = get_top_K_docnos(sorted_doc_scores)
        
        ###
        top_K_docs_text = {docno: preprocessed_files_text[docno] for docno in top_K_docnos}
        query_text = preprocessed_queries_text[num]
        sorted_doc_scores = bert_process(num, query_text, top_K_docs_text)
        ###
        
        ### OR ###
        
        ###
        # top_K_docs_tokens = {docno: preprocessed_queries_tokens[docno] for docno in top_K_docnos}
        # query_tokens = preprocessed_queries_tokens[num]
        # sorted_doc_scores = doc2vec_process(num, query_tokens, top_K_docs_tokens)
        ###
        
        write_results_into_text_file(num, sorted_doc_scores, run_name='bm25_st')
        
    print(f'Running trec_eval for file result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
                
if __name__ == "__main__":
    main()
