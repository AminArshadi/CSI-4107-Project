from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pprint import pprint
import nltk
import re
import json

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
    
    for doc in docs:
        docno, content = doc['docno'], doc['content']
        tokens = nltk.word_tokenize(content)
        tokens = [token for token in tokens if token.isalpha() and (token not in stop_words)]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        result[docno] = stemmed_tokens
        
        print(f'Preprocessing of document {docno} finished.')
                
    print(f'Preprocessing of all documents in {file_path} finished.')
    return result
    
def preprocess_queries(file_path):
    queries = parse_queries(file_path)
    
    result = {}
    
    for query in queries:
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
