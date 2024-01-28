import os
import re
import nltk
import json

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
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

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_documents(file_path):
    docs = parse_document(file_path)
    
    result = {}
    
    for doc in docs: # docs[:1] -> docs
        docno, content = doc['docno'], doc['content']
        tokens = nltk.word_tokenize(content)
        tokens = [token for token in tokens if token.isalpha() and (token not in stop_words)]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        result[docno] = stemmed_tokens
                
    print(f'Preprocessing of all documents in {file_path} finished.')
    return result
    
def preprocess_queries(file_path):
    queries = parse_queries(file_path)
    
    result = {}
    
    for query in queries: # queries[:1] -> queries
        num, content = query['num'], query['content']
        
        tokens = nltk.word_tokenize(content)
        tokens = [token for token in tokens if token.isalpha() and (token not in stop_words)]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        result[num] = stemmed_tokens
    
    print(f'Preprocessing of all queries finished.')
        
    return result

def get_inverted_index(doc_tokens_dict):
    inverted_index = {}
    for docno, stemmed_tokens in doc_tokens_dict.items():
        for token in stemmed_tokens:
            if token not in inverted_index:
                inverted_index[token] = [docno]
            elif docno not in inverted_index[token]:
                inverted_index[token].append(docno)
    return inverted_index

def save_to_json(data, file_name):
    with open(file_name, "w") as file:
        json.dump(data, file)
    
def load_from_json(file_name):
    with open(file_name, "r") as file:
        return json.load(file)

def main():    
    # preprocessed_files = {}
    # directory = './documents'
    
    # for filename in os.listdir(directory):
    #     file_path = os.path.join(directory, filename)
    #     preprocessed_files.update(preprocess_documents(file_path))
        
    # inverted_index = get_inverted_index(preprocessed_files)
    
    # save_to_json(preprocessed_files, "./saved_preprocessed_files.json")
    # save_to_json(inverted_index, "./saved_inverted_index.json")
    
    ##############################################################################################################################
    
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
    
    bm25 = BM25Okapi(useful_preprocessed_files.values())
    
    for num, query_tokens in preprocessed_queries.items():
        docnos = preprocessed_files.keys()
        scores = bm25.get_scores(query_tokens)
        sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
        
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            print(f'{num} Q0 {docno} {rank + 1} {score} run_name_{rank}')
    
if __name__ == "__main__":
    main()
