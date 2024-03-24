import os

from utils.main import *

def main():
    preprocessed_files_tokens, preprocessed_files_text = {}, {}
    directory = './documents'
    
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures_tokens, futures_text = [], []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            futures_tokens.append(executor.submit(preprocess_documents, file_path, type='tokens'))
            futures_text.append(executor.submit(preprocess_documents, file_path, type='text'))
        
        for future in futures_tokens:
            preprocessed_files_tokens.update(future.result())
            
        for future in futures_text:
            preprocessed_files_text.update(future.result())
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'Finished preprocessing all documents in {duration} minutes.')
    
    start_time = time.time()
    print('Making the inverted index ...')
    inverted_index = get_inverted_index(preprocessed_files_tokens, type='tokens')
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'Finished making the inverted index in {duration} minutes.')
    
    print('Writing into saved_preprocessed_files_tokens.json ...')
    save_to_json(preprocessed_files_tokens, "./saved_preprocessed_files_tokens.json")
    print('Done.')
    
    print('Writing into saved_preprocessed_files_text.json ...')
    save_to_json(preprocessed_files_text, "./saved_preprocessed_files_text.json")
    print('Done.')
    
    print('Writing into saved_inverted_index.json ...')
    save_to_json(inverted_index, "./saved_inverted_index.json")
    print('Done.')
    
if __name__ == "__main__":
   main()
