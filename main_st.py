from utils.main import *
from sentence_transformers import SentenceTransformer, util

ST = SentenceTransformer('all-MiniLM-L6-v2')
# ST = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
# ST = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def bert_process(num, query_text, useful_preprocessed_files):
    '''
    Processes a query and a collection of documents using BERT embeddings to compute cosine similarity scores.
    Parameters:
        num (int): The query number for identification.
        query_text (str): The text of the query to be processed.
        useful_preprocessed_files (dict): A dictionary with document identifiers as keys and their preprocessed text as values.
    Returns:
        list of tuples: Each tuple contains a document identifier and its corresponding similarity score with the query, sorted in descending order of similarity.
    Note:
        Utilizes the Sentence Transformers library to encode both the query and documents into embeddings before computing cosine similarity. Requires preloading the desired BERT model ('all-MiniLM-L6-v2', 'roberta-large-nli-stsb-mean-tokens', or 'distilbert-base-nli-stsb-mean-tokens') and the `util` module for cosine similarity calculation.
    '''
    print(f'Getting the results for query {num} ...')
    start_time = time.time()
    
    document_embeddings = ST.encode(list(useful_preprocessed_files.values()), convert_to_tensor=True)
    query_embedding = ST.encode(query_text, convert_to_tensor=True)
    
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)
    cosine_scores = cosine_scores.cpu().numpy().flatten().tolist() # Convert to list of floats
    
    doc_scores = list(zip(useful_preprocessed_files.keys(), cosine_scores))
    sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'Finished in {duration} minutes.')
    
    return sorted_doc_scores

def compute_scores_for_query(args):
    '''
    Computes similarity scores for a single query against a set of useful preprocessed files using BERT embeddings.
    Parameters:
        args (tuple): A tuple containing the query number, query tokens, preprocessed files tokens, and the inverted index.
    Returns:
        tuple: A tuple consisting of the query number and a list of tuples, each containing a document identifier and its similarity score to the query, sorted in descending order.
    Note:
        This function is designed to work within a concurrent execution environment, processing individual queries in parallel for efficiency.
    '''
    num, query_tokens, preprocessed_files_tokens, inverted_index = args
    useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='text')
    sorted_doc_scores = bert_process(num, query_tokens, useful_preprocessed_files)
    return num, sorted_doc_scores

def compute_scores_for_chunk(chunk):
    '''
    Processes a chunk of queries, computing similarity scores for each query in the chunk using BERT embeddings.
    Parameters:
        chunk (list of tuples): A list where each tuple contains a query's number, tokens, preprocessed files tokens, and inverted index.
    Returns:
        list: A list of results for each query in the chunk, where each result is a tuple containing the query number and its sorted document scores.
    Note:
        Utilizes `compute_scores_for_query` for each query in the chunk, facilitating parallel processing of multiple queries for increased efficiency.
    '''
    results_for_chunk = [compute_scores_for_query(args) for args in chunk]
    return results_for_chunk

def main():
    '''
    Main function for processing queries against a corpus using BERT embeddings to find the most relevant documents. 
    This function loads preprocessed document text and an inverted index, preprocesses queries, filters documents relevant to each query, ranks documents based on similarity scores obtained from BERT processing, and writes the results to a text file. Finally, it evaluates the results using a TREC evaluation script.
    Note:
        Assumes utility functions for loading data, preprocessing queries, filtering documents, writing results, and evaluating performance are imported from 'utils.main'.
    '''
    preprocessed_files_text = load_from_json("saved_preprocessed_files_text.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries', type='text')
    
    all_doc_scores = {}
        
    all_data = [(num, query_tokens, preprocessed_files_text, inverted_index) for num, query_tokens in preprocessed_queries.items()]
    chunk_size = 10
    query_chunks = list(chunks(all_data, chunk_size))
    with ProcessPoolExecutor() as executor:
        for chunk_result in executor.map(compute_scores_for_chunk, query_chunks):
            for num, sorted_doc_scores in chunk_result:
                all_doc_scores[int(num)] = sorted_doc_scores
    
    sorted_all_doc_scores = sorted(all_doc_scores.items(), key=lambda x: x[0])
    for num, sorted_doc_scores in sorted_all_doc_scores:
        write_results_into_text_file(num, sorted_doc_scores, run_name='st')
        
    print(f'Running trec_eval for result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
        
if __name__ == "__main__":
    main()
