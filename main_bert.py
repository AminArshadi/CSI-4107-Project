# python pre_run.py text
# python main_bert.py

import os
import torch
import time

from utils.main import *
from semantic_text_similarity.models import WebBertSimilarity


# BERT = WebBertSimilarity(device='cpu')
BERT = WebBertSimilarity(device='cpu', batch_size=32)


def bert_process(num, query_text, useful_preprocessed_files):
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# from transformers import AutoTokenizer, AutoModel
    
    
    
    
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModel.from_pretrained("bert-base-uncased")
    
    # input_ids = []
    # attention_masks = []
        
    # for text in preprocessed_files.values():
        
    #     encoded_inputs = tokenizer(text, padding=False, truncation=True) # Tokenize texts and prepare input tensors
    #     input_ids.append(encoded_inputs['input_ids'])
    #     attention_masks.append(encoded_inputs['attention_mask'])
    
    # max_length = max([len(ids) for ids in input_ids])
        
    # input_ids = [ids + [tokenizer.pad_token_id]*(max_length-len(ids)) for ids in input_ids]
    # attention_masks = [mask + [0]*(max_length-len(mask)) for mask in attention_masks]

    # input_ids = torch.tensor(input_ids)
    # attention_masks = torch.tensor(attention_masks)
        
    # outputs = model(input_ids=input_ids, attention_mask=attention_masks) # Feed inputs to model
    
    # embeddings = outputs.last_hidden_state # Extract embeddings
    
    # cls_embeddings = embeddings[:, 0, :] # For the purpose of neural ranking
    
    # print('Writing into cls_embeddings.json ...')
    # save_to_json(cls_embeddings, "./cls_embeddings.json")
        
if __name__ == "__main__":
    main()
