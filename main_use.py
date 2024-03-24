import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from utils.main import *

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def use_process(query_text, useful_preprocessed_files):    
    document_embeddings = embed(list(useful_preprocessed_files.values()))
    query_embedding = embed([query_text])

    cosine_similarities = np.inner(query_embedding, document_embeddings)[0]

    doc_scores = list(zip(useful_preprocessed_files.keys(), cosine_similarities))
    sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_doc_scores

def main():
    preprocessed_files_text = load_from_json("saved_preprocessed_files_text.json")
    inverted_index = load_from_json("saved_inverted_index.json")
    
    preprocessed_queries = preprocess_queries('./queries', type='text')
    
    for num, query_text in preprocessed_queries.items():
        useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_text, inverted_index, query_text, type='text')
        sorted_doc_scores = use_process(num, query_text, useful_preprocessed_files)
        write_results_into_text_file(num, sorted_doc_scores, run_name='st')
        
    print(f'Running trec_eval for result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', './trec_eval/results/result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
        
if __name__ == "__main__":
    main()
