from utils.main import *
import os

def main():
    preprocessed_files = {}
    directory = './documents'
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        preprocessed_files.update(preprocess_documents(file_path))
    
    print('Making the inverted index ...')
    inverted_index = get_inverted_index(preprocessed_files)
    print('Done.')
    
    print('Writing into saved_preprocessed_files.json ...')
    save_to_json(preprocessed_files, "./saved_preprocessed_files.json")
    print('Done.')
    
    print('Writing into saved_inverted_index.json ...')
    save_to_json(inverted_index, "./saved_inverted_index.json")
    print('Done.')
    
if __name__ == "__main__":
    main()
