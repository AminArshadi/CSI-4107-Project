import spacy
import re

def read_doc(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def parse_document(file_path):
    content = read_doc(file_path)

    documents = []
    doc_splits = content.split('<DOC>')
    
    for doc in doc_splits:
        if doc.strip():
            
            docno = re.search('<DOCNO>(.*?)</DOCNO>', doc, re.DOTALL)
            fileid = re.search('<FILEID>(.*?)</FILEID>', doc, re.DOTALL)
            headline = re.search('<HEAD>(.*?)</HEAD>', doc, re.DOTALL)
            text = re.search('<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            
            docno = docno.group(1).strip() if docno else None
            fileid = fileid.group(1).strip() if fileid else None
            headline = headline.group(1).strip() if headline else None
            text = text.group(1).strip() if text else None
            
            info = {
                'docno':docno,
                'fileid':fileid,
                'headline':headline,
                'text':text,}

            documents.append(info)

    return documents

def tokenize_and_lemmatize(text):
    # Load the English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Tokenize and lemmatize
    tokens = [(token.text, token.lemma_) for token in doc]

    return tokens


def main():
    text = "Apples are delicious"
    result = tokenize_and_lemmatize(text)
    print(result)
    
main()
