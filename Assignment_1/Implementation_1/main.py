import os
import re
import nltk
import json
import spacy

from rank_bm25 import BM25Okapi
from pprint import pprint


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def parse_document(file_path):
    content = read_file(file_path)

    documents = []
    doc_splits = content.split('<DOC>')
    
    for doc in doc_splits:
        if doc.strip():
            
            docno = re.search('<DOCNO>(.*?)</DOCNO>', doc, re.DOTALL)
            headline = re.search('<HEAD>(.*?)</HEAD>', doc, re.DOTALL)
            text = re.search('<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            
            docno = docno.group(1).strip() if docno else None
            headline = headline.group(1).strip() if headline else ''
            text = text.group(1).strip() if text else ''
            
            info = {
                'docno':docno,
                'content':headline + text
                }

            documents.append(info)
            
    return documents

def parse_queries(file_path):
    content = read_file(file_path)

    queries = []
    query_splits = content.split('<top>')
    
    for query in query_splits:
        if query.strip():
            
            num = re.search('<num>(.*?)<title>', query, re.DOTALL)
            title = re.search('<title>(.*?)<desc>', query, re.DOTALL)
            desc = re.search('<desc>(.*?)<narr>', query, re.DOTALL)
            
            num = num.group(1).strip() if num else None
            title = title.group(1).strip() if title else ''
            desc = desc.group(1).strip() if desc else ''
            
            info = {
                'num':num,
                'content':title + desc
                }

            queries.append(info)

    return queries

def tokenize(text):
    return nltk.word_tokenize(text)

def lemmatize(token):
    nlp = spacy.load("en_core_web_sm") # Load the English model
    processed_token = nlp(token.lower()) # Process the text
    
    # Lemmatize and remove stop words, puncuations, and numbers.
    tokens = [token.lemma_ for token in processed_token if (not token.is_stop) and (not token.is_punct) and (not token.like_num) and token.is_alpha]
            
    return tokens

def lemmatize_tokens(tokens):
    lemmatized_tokens = []
    for token in tokens:
        lemmatized_token = lemmatize(token)
        if len(lemmatized_token) != 0:
            lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

def preprocess_documents(file_path):
    docs = parse_document(file_path)
    
    lemmatized_tokenized_docs, docnos = [], []
    
    for doc in docs: # docs[:1] -> docs
        content, docno = doc['content'], doc['docno']
        tokens = tokenize(content)
        lemmatized_tokens = lemmatize_tokens(tokens)
        docnos.append(docno)
        flattened_lemmatized_tokens = [token for sublist in lemmatized_tokens for token in sublist] # flattening a 2-D list
        lemmatized_tokenized_docs.append(flattened_lemmatized_tokens)
        
        print(f'Preprocessing of document {docno} finished.')
    
    if os.path.exists("./saved_info.json"): # If saved_info.json already exists
        with open("./saved_info.json", "r") as file:
            existing_info = json.load(file)
        existing_info['docnos'].extend(docnos)
        existing_info['lemmatized_tokenized_docs'].extend(lemmatized_tokenized_docs)
        info = existing_info
        
    else: # If saved_info.json doesn't exist
        info = {
            'docnos': docnos,
            'lemmatized_tokenized_docs': lemmatized_tokenized_docs
        }
    
    with open("./saved_info.json", "w") as file:
        json.dump(info, file)
        
    print(f'Preprocessing of all documents in {file_path} finished.')
        
    return
    
def preprocess_queries(file_path):
    queries = parse_queries(file_path)
    
    nums, lemmatized_tokenized_queries = [], []
    
    for query in queries[:1]: # queries[:1] -> queries
        tokens = tokenize(query['content'])
        lemmatized_tokens = lemmatize_tokens(tokens)
        
        nums.append(query['num'])
        
        flattened_lemmatized_tokens = [token for sublist in lemmatized_tokens for token in sublist] # flattening a 2-D list
        lemmatized_tokenized_queries.append(flattened_lemmatized_tokens)
        
    return nums, lemmatized_tokenized_queries

def main():
    
    directory = './documents'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        preprocess_documents(file_path)
        
    ##############################################################################################################################
    
    with open("saved_info.json", "r") as file:
        info = json.load(file)
    
    docnos, lemmatized_tokenized_docs = info['docnos'], info['lemmatized_tokenized_docs']
    
    bm25 = BM25Okapi(lemmatized_tokenized_docs)
        
    nums, lemmatized_tokenized_queries = preprocess_queries('./queries')
    
    for num, lemmatized_tokenized_querie in zip(nums, lemmatized_tokenized_queries):
        scores = bm25.get_scores(lemmatized_tokenized_querie)
        
        sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
        
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            print(f'{num} Q0 {docno} {rank + 1} {score} run_name')
    
if __name__ == "__main__":
    main()
