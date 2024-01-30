from utils.main import *

def main():
    directory = './documents'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        preprocess_documents(file_path)
    
if __name__ == "__main__":
    main()
