import spacy

def read_doc(file_path):
    with open(file_path, 'r') as file:
        return file.read()

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
