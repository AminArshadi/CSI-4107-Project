from utils.main import *
from rank_bm25 import BM25Okapi, BM25Plus

def bm25_proccess(query_tokens, useful_preprocessed_files, k1=2, b=0.66):
    '''
    Processes a query against a collection of documents using the BM25+ algorithm to compute relevance scores.
    Parameters:
        query_tokens (list of str): The tokenized query.
        useful_preprocessed_files (dict): A dictionary with document identifiers as keys and their tokenized content as values.
        k1 (float, optional): The k1 parameter in the BM25+ algorithm, controlling term frequency scaling. Defaults to 2.
        b (float, optional): The b parameter in the BM25+ algorithm, controlling document length normalization. Defaults to 0.66.
    Returns:
        list of tuples: Each tuple contains a document identifier and its BM25+ score relative to the query, sorted in descending order of score.
    Note:
        Utilizes the `rank_bm25.BM25Plus` class for the BM25+ algorithm. The function is flexible to changes in the BM25+ parameters for experimentation.
    '''
    bm25 = BM25Plus(useful_preprocessed_files.values(), k1=k1, b=b)
    
    docnos = useful_preprocessed_files.keys()
    scores = bm25.get_scores(query_tokens)
    sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True)
    return sorted_doc_scores
        
def compute_scores_for_query(args):
    '''
    Computes and ranks documents based on their relevance to a query using the BM25+ algorithm.
    Parameters:
        args (tuple): Contains the query number, query tokens, preprocessed files tokens, inverted index, and BM25+ parameters (k1, b).
    Returns:
        tuple: The query number and a sorted list of document scores.
    Note:
        This function is tailored for processing within a parallel execution framework, enabling efficient handling of multiple queries.
    '''
    num, query_tokens, preprocessed_files_tokens, inverted_index, K1, B = args
    useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='tokens')
    sorted_doc_scores = bm25_proccess(query_tokens, useful_preprocessed_files, k1=K1, b=B)
    return num, sorted_doc_scores

def compute_scores_for_chunk(chunk):
    '''
    Processes a chunk of queries, computing and ranking documents for each query based on BM25+ scores.
    Parameters:
        chunk (list of tuples): A list of arguments for `compute_scores_for_query`, each corresponding to a single query.
    Returns:
        list: A list of results for each query in the chunk, facilitating batch processing.
    Note:
        Designed for use with concurrent processing to enhance efficiency in large-scale document ranking tasks.
    '''
    results_for_chunk = [compute_scores_for_query(args) for args in chunk]
    return results_for_chunk
    
def main():
    '''
    Main function for executing the information retrieval process using the BM25 algorithm.
    Loads preprocessed document tokens and an inverted index, processes queries, computes relevance scores for documents per query using BM25, and writes the ranked results to a file. Finally, it runs the TREC evaluation script to assess performance.
    The process involves:
        - Loading necessary data from JSON.
        - Preprocessing queries.
        - Chunking queries for parallel processing.
        - Computing BM25 scores for each query against useful documents.
        - Writing results and evaluating with TREC eval.
    Note:
        This function organizes the workflow for document ranking with BM25, leveraging parallel processing for efficiency and scalability.
    '''
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
