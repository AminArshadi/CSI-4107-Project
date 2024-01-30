from utils.main import *

def main():
    with open("saved_info.json", "r") as file:
        info = json.load(file)
    
    docnos, lemmatized_tokenized_docs = info['docnos'], info['lemmatized_tokenized_docs']
    
    bm25 = BM25Okapi(lemmatized_tokenized_docs)
        
    nums, lemmatized_tokenized_queries = preprocess_queries('./queries')
    
    result_file = open('Results.txt', 'w')
    
    for num, lemmatized_tokenized_querie in zip(nums, lemmatized_tokenized_queries):
        scores = bm25.get_scores(lemmatized_tokenized_querie)
        
        sorted_doc_scores = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True) # Ranking documents based on their scores
        
        for rank, (docno, score) in enumerate(sorted_doc_scores):
            result_file.write(f'{num} Q0 {docno} {rank + 1} {score} run_name_{rank}')
    
    result_file.close()
    
if __name__ == "__main__":
    main()
