from utils.main import *
from semantic_text_similarity.models import WebBertSimilarity

BERT = WebBertSimilarity(device='cpu', batch_size=32)

def bert_process(num, query_text, useful_preprocessed_files):
    '''
    Computes document relevance scores for a given query using BERT embeddings and writes the results to a file.
    Parameters:
        num (int): The query number for identification purposes.
        query_text (str): The text of the query.
        useful_preprocessed_files (dict): A dictionary mapping document identifiers to their corresponding preprocessed text.
    Returns:
        None: This function writes the sorted document scores directly to a result file and does not return any value.
    Note:
        Utilizes the WebBertSimilarity model for computing similarity scores between the query and each document in the corpus. The function logs processing duration and writes results to a specified result file.
    '''
    print(f'Getting the results for query {num} ...')
    start_time = time.time()
    batch_pairs = [(query_text, doc) for doc in useful_preprocessed_files.values()] # Prepare batch pairs for each query and all preprocessed documents
    similarity_scores = BERT.predict(batch_pairs) # Compute similarity scores in batches
    doc_scores = list(zip(useful_preprocessed_files.keys(), similarity_scores))
    
    sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'Finished in {duration} minutes.')
    
    print(f'Adding the results of query {num} to the result.txt file ...')
    with open('./trec_eval/results/result.txt', 'a') as result_file:
        for rank, (docno, score) in enumerate(sorted_doc_scores, start=1):
            result_file.write(f'{num} Q0 {docno} {rank} {score} run_name\n')
    print('Done.')

def main():
    '''
    The main execution function to process all queries using BERT-based similarity, rank documents, and evaluate the system.
    It loads preprocessed document texts and an inverted index, processes each query to compute similarity scores with documents using BERT embeddings, and appends the results to a results file. Finally, it runs a TREC evaluation to assess the performance.
    Note:
        This function assumes the presence of preloaded data, a set of preprocessed queries, and utility functions for loading data, processing queries, and evaluating results.
    '''
    preprocessed_files_text = load_from_json("saved_preprocessed_files_text.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries', type='text')
    
    for num, query_text in preprocessed_queries.items():
        useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_text, inverted_index, query_text, type='text')
        bert_process(num, query_text, useful_preprocessed_files)
        
    print(f'Running trec_eval for result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
        
if __name__ == "__main__":
    main()
