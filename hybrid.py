from utils.main import *

import os

# 0.4288

FILES_WEIGHTS = {
    'all-MiniLM-L6-v2.txt': 2.2,
    'bm25.txt': 4,
}

TOTAL_WEIGHTS = sum(FILES_WEIGHTS.values())

DIRECTORY = './trec_eval/results'

def main():
    new_results = [{} for _ in range(50)]
    
    for file_name, weight in FILES_WEIGHTS.items():
        file_path = os.path.join(DIRECTORY, file_name)
        
        with open(file_path, "r") as file:
            data = file.read()
            data_list = data.split('\n')
            data_list = list(filter(None, data_list))
            
        results = [{} for _ in range(50)]
        
        for result in data_list:
            result_list = result.strip().split(' ')
            
            num = result_list[0]
            docno = result_list[2]
            score = result_list[4]
            
            results[int(num) - 1][docno] = float(score)
            
        normalized_weighted_results = []
        
        for result in results:
            scores = result.values()
            min_score = min(scores)
            max_score = max(scores)
            
            for docno, score in result.items():
                normalized_score = (score - min_score) / (max_score - min_score) if max_score != min_score else 0
                normalized_weighted_score = weight * normalized_score
                result[docno] = normalized_weighted_score
                
            normalized_weighted_results.append(result)
        
        for i in range(50):
            for docno, score in normalized_weighted_results[i].items():
                if docno in new_results[i]:
                    new_results[i][docno] += score
                else:
                    new_results[i][docno] = score
    
    for i in range(50):
        new_results[i] = {docno: score / TOTAL_WEIGHTS for docno, score in new_results[i].items()}
    
    for i in range(50):
        sorted_doc_scores = sorted(new_results[i].items(), key=lambda x: x[1], reverse=True)
        with open('./trec_eval/results/hybrid_result.txt', 'a') as result_file:
            for rank, (docno, score) in enumerate(sorted_doc_scores, start=1):
                result_file.write(f'{i + 1} Q0 {docno} {rank} {score} hybrid\n')
    
    print(f'Running trec_eval for hybrid_result.txt ...')
    executable = ['./trec_eval/trec_eval', './trec_eval/qrel.txt', f'./trec_eval/results/hybrid_result.txt']
    output = run_trac_eval(executable)
    print(output)
    print('Done.')
    
if __name__ == "__main__":
    main()
