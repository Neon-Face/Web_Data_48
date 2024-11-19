# ##Remove Stop words and Perform Text Normalization

import nltk
from nltk.corpus import stopwords
import en_core_web_sm
from nltk.stem import PorterStemmer

# Download lists
def necessary_download():
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


# Remove stopwords using NLTK
def stop_words(stops, tokens):        
    filtered_sentence = [] 
    
    # remove stop words from given tokens
    for w in tokens: 
        if w not in stops: 
            filtered_sentence.append(w) 

    return filtered_sentence


# Words Lemmatization
def lemma_words(stopwords): # sentences without stopwords
    text = stopwords #for simplication
    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(text))
    
    lemma_word = [] 
    for token in doc:
        lemma_word.append(token.lemma_)
    
    return lemma_word


# Words Stemming
def stem(stopwords):
    text = stopwords
    stem_words = []
    ps =PorterStemmer()

    for w in text:
        rootWord=ps.stem(w)
        stem_words.append(rootWord)

    return stem_words