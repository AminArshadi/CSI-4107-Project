from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from pprint import pprint
import nltk
import spacy
import re
import json
import contractions

STOP_WORDS = {
    'cos', 'noting', 'eighty', 'obtaining', 'soon', 'av', 'taken', 'usefully', 'seeing', 'this', 'ye', 'important', 'den', 'hasn', 'ends', 'something', 'an',
    'every', 'forward', 'third', 'least', 'afterwards', 'dans', 'selves', 'high', 'said', 'le', 'other', 'sometime', 'nearly', 'l', 'sprung', 'such', 'per',
    'note', 'won', 'ii', 'next', 'farthest', 'useful', 'whichsoever', 'kg', 'certainly', 's', 'y', 'does', 'apparently', 'his', 'really', 'plus', 'sobre',
    'underneath', 'own', 'last', 'speek', 'por', 'wherever', 'giving', 'come', 'especially', 'spat', 'at', 'end', 'we', 'am', 'see', 'h', 'ot', 'contrariwise',
    'auf', 'simple', 'ja', 'hundred', 'didn', 'and', 'ohne', 'who', 'needing', 'fuer', 'quite', 'my', 'everyone', 'wow', 'whichever', 'large', 'si', 'latter',
    'wherewith', 'chooses', 'himself', 'larger', 'nobody', 'some', 'caption', 'might', 'during', 'frequently', 'whom', 'seem', 'xauthor', 'anything', 'wilt',
    'd', 'made', 'nas', 'now', 'also', 'cu', 'around', 'including', 'anyway', 'whereinto', 'do', 'within', 'probably', 'becoming', 'smote', 'successfully',
    'whereupon', 'half', 'included', 'insomuch', 'usually', 'never', 'seven', 'spake', 'el', 'meantime', 'ten', 'mit', 'then', 'week', 'furthermore', 'between',
    'fact', 'journals', 'fra', 'single', 'sulla', 'cannot', 'low', 'themselves', 'best', 'og', 'upward', 'thereabout', 'thou', 'ueber', 'excepted', 'elsewhere',
    'down', 'tis', 'inc', 'al', 'somewhere', 'die', 'whosoever', 'din', 'foer', 'thru', 'obtains', 'shouldn', 'les', 'has', 'whereof', 'inside', 'of', 'includes',
    'clearly', 'old', 're', 'particular', 'without', 'predominantly', 'sending', 'degli', 'aus', 'pour', 'thus', 'chosen', 'i', 'sleeping', 'six', 'spit', 'however',
    'your', 'howsoever', 'things', 'in', 'unless', 'whatever', 'both', 'dos', 'kai', 'til', 'followed', 'm', 'many', 'somebody', 'two', 'double', 'up', 'the',
    'necessarily', 'mug', 'aren', 'sur', 'providing', 'though', 'overall', 'de', 'likely', 'go', 'generally', 'did', 'dual', 'going', 'r', 'say', 'several',
    'thousand', 'mine', 'beginning', 'as', 'hindmost', 'all', 'wherein', 'long', 'sixty', 'prompt', 'substantially', 'fifty', 'co', 'hardly', 'whereto', 'show',
    'it', 'maybe', 'according', 'why', 'against', 'whew', 'whence', 'takes', 'ok', 'wasn', 'sing', 'haven', 'staves', 'he', 'yourselves', 'few', 'necessary',
    'none', 'how', 'ourselves', 'sideways', 'her', 'out', 'becomes', 'different', 'forty', 'more', 'even', 'need', 'ltd', 'everywhere', 'possible', 'thousands',
    'en', 'worst', 'small', 'simply', 'already', 'gave', 'v', 'while', 'always', 'k', 'otherwise', 'nor', 'wheresoever', 'had', 'me', 'journal', 'di', 'la', 'n',
    'xnote', 'seemed', 'often', 'excluding', 'kind', 'nowhere', 'nella', 'therefor', 'therein', 'lowest', 'new', 'alone', 'seventy', 'facts', 'users', 'isn',
    'immediately', 'excluded', 'just', 'provided', 'shall', 'strongly', 'no', 'stop', 'beside', 'thyself', 'un', 'sang', 'being', 'began', 'spits', 'worse',
    'further', 'keep', 'because', 'us', 'exclude', 'hereabouts', 'whereunto', 'aside', 'actually', 'excepts', 'formerly', 'ac', 'shown', 'weren', 'vs',
    'nevertheless', 'nonetheless', 'can', 'doesn', 'ie', 'furthest', 'unlikely', 'using', 'whensoever', 'whole', 'spoke', 'yours', 'enough', 'af', 'three',
    'found', 'deren', 'much', 'highest', 'somewhat', 'ours', 'should', 'whenever', 'came', 'und', 'uit', 'whereby', 'whomever', 'dei', 'j', 'less', 'notwithstanding',
    'sends', 'sleep', 'weeks', 'particularly', 'later', 'na', 'het', 'xother', 'given', 'beyond', 'whither', 'ending', 'aux', 'whoever', 'sung', 'u', 'better',
    'em', 'della', 'inasmuch', 'a', 'various', 'thy', 'since', 'nel', 'thereby', 'or', 'over', 'another', 'whereas', 'ought', 'da', 'gone', 'through', 'thereafter',
    'provide', 'exclusive', 'hereby', 'their', 'za', 'spitting', 'excepting', 'these', 'get', 'others', 'for', 'anybody', 'anywhere', 'seems', 'most', 'well',
    'thing', 'above', 'taking', 'you', 'largest', 'mainly', 'thence', 'nach', 'haedly', 'besides', 'eight', 'unable', 'among', 'similarly', 'slightly', 'hath',
    'when', 'very', 'whomsoever', 'durch', 'hereupon', 'notes', 'nowadays', 'front', 'indeed', 'don', 'send', 'seeming', 'thereupon', 'together', 'almost',
    'second', 'albeit', 'save', 'q', 'gives', 'med', 'vom', 'ff', 'little', 'promptly', 'las', 'needs', 'still', 'one', 'hither', 'moreover', 'showed', 'voor',
    'five', 'via', 'sleeps', 'apart', 'nos', 'before', 'recent', 'ugh', 'behind', 'rather', 'etc', 'somehow', 'forth', 'van', 'para', 'yet', 'toward', 'pours',
    'indoors', 'trillions', 'be', 'that', 'xcal', 'here', 'km', 'au', 'lower', 'off', 'what', 'below', 'exception', 'any', 'f', 'near', 'sui', 'which', 'adj',
    'shalt', 'making', 'makes', 'halves', 'needed', 'round', 'thereof', 'wherefrom', 'wide', 'seen', 'del', 'therefore', 'obtained', 'dem', 'those', 'about',
    'certain', 'million', 'tes', 'yes', 'them', 'x', 'et', 'owing', 'tot', 'largely', 'user', 'henceforth', 'te', 'speeks', 'throughout', 'sees', 'recently',
    'doing', 'four', 'cf', 'its', 'him', 'seldom', 'whether', 'use', 'par', 'clear', 'herself', 'must', 'wherefore', 'everybody', 'herein', 'whereon', 'longer',
    'let', 'chose', 'w', 'neither', 'noone', 'nope', 'och', 'los', 'they', 'done', 'sent', 't', 'inward', 'plenty', 'same', 'tou', 'keeping', 'everything', 'day',
    'significant', 'respectively', 'whose', 'been', 'z', 'along', 'theirs', 'ever', 'anyone', 'spoken', 'regardless', 'finally', 'latterly', 'ways', 'll', 'short',
    'zu', 'getting', 'mostly', 'slew', 'im', 'couldn', 'e', 'into', 'con', 'ihre', 'having', 'als', 'amongst', 'relatively', 'include', 'upon', 'away', 'om', 'than',
    'thenceforth', 'used', 'pouring', 'on', 'eg', 'first', 'yu', 'ready', 'would', 'avec', 'general', 'where', 'hereafter', 'way', 'begin', 'begins', 'possibly',
    'outside', 'there', 'mr', 'once', 'thereabouts', 'previously', 'became', 'each', 'not', 'towards', 'if', 'are', 'bei', 'gets', 'hundreds', 'hence', 'from',
    'widely', 'unlike', 'due', 'obtain', 'again', 'after', 'following', 'noted', 'p', 'kinds', 'choosing', 'von', 'comes', 'pro', 'by', 'c', 'twenty', 'but',
    'become', 'past', 'captions', 'choose', 'farther', 'ms', 'instead', 'miss', 'shows', 'could', 'yourself', 'got', 'similar', 'saw', 'showing', 'trillion', 'myself',
    'whereabouts', 'uses', 'es', 'meanwhile', 'ses', 'das', 'whereat', 'times', 'former', 'beforehand', 'thrice', 'accordingly', 'ou', 'des', 'will', 'follows',
    'someone', 'is', 'till', 'nine', 'like', 'b', 'namely', 'hers', 'thirty', 'have', 'either', 'until', 'ed', 'was', 'briefly', 'take', 'our', 'follow', 'so',
    'sprang', 'went', 'g', 'hereto', 'were', 'across', 've', 'higher', 'too', 'far', 'to', 'excludes', 'thereto', 'du', 'she', 'want', 'lest', 'ended', 'xsubj',
    'give', 'may', 'under', 'ze', 'dost', 'kept', 'onto', 'self', 'wouldn', 'whilst', 'ad', 'except', 'great', 'poured', 'time', 'slept', 'whereafter', 'good',
    'anyhow', 'arise', 'year', 'delle', 'supposing', 'nothing', 'make', 'please', 'o', 'perhaps', 'only', 'itself', 'mrs', 'thereon', 'with', 'although',
    'sometimes', 'ninety', 'greater', 'onceone', 'zum', 'else', 'hast', 'der', 'whatsoever', 'provides', 'billion', 'yipee', 'canst', 'thee'
    }

STOP_WORDS |= set(stopwords.words('english'))
STEMMER = PorterStemmer()
NLP = spacy.load("en_core_web_sm")

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

def preprocess(content):
    content = contractions.fix(content.lower()) # example: can't -> cannot
    tokens = nltk.word_tokenize(content)
    tokens = [token for token in tokens if token.isalpha() and (token not in STOP_WORDS)]
    
    ###
    lemma_tokens = []
    for token in tokens:
        processed_token = NLP(token)
        lemma_token = [token.lemma_ for token in processed_token]
        lemma_tokens.extend(lemma_token)
    return lemma_tokens
    ###
    
    ### OR ###
    
    ###
    # stemmed_tokens = [STEMMER.stem(token) for token in tokens]
    # return stemmed_tokens
    ###

def preprocess_documents(file_path):
    docs = parse_document(file_path)
    
    result = {}
    
    for doc in docs:
        docno, content = doc['docno'], doc['content']
        stemmed_tokens = preprocess(content)
        result[docno] = stemmed_tokens
        
        print(f'Preprocessing of document {docno} finished.')
                
    print(f'Preprocessing of all documents in {file_path} finished.')
    return result
    
def preprocess_queries(file_path):
    queries = parse_queries(file_path)
    
    result = {}
    
    for query in queries:
        num, content = query['num'], query['content']
        stemmed_tokens = preprocess(content)
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
