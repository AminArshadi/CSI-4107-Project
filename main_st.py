# python pre_run.py text
# python main_st.py

import os
import time
import torch

from utils.main import *
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer


ST = SentenceTransformer('all-MiniLM-L6-v2') # 0.2952
# ST = SentenceTransformer('roberta-large-nli-stsb-mean-tokens') # 0.0999
# ST = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens') # 0.0882

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

def expand_query_with_gpt2(query_text):
    input_ids = gpt_tokenizer.encode(query_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    outputs = gpt_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150)
    expanded_query = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return expanded_query

def bert_process(num, query_text, useful_preprocessed_files):
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
    num, query_tokens, preprocessed_files_tokens, inverted_index = args
    useful_preprocessed_files = get_useful_preprocessed_files(preprocessed_files_tokens, inverted_index, query_tokens, type='text')
    sorted_doc_scores = bert_process(num, query_tokens, useful_preprocessed_files)
    return num, sorted_doc_scores

def compute_scores_for_chunk(chunk):
    results_for_chunk = [compute_scores_for_query(args) for args in chunk]
    return results_for_chunk

def main():
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
