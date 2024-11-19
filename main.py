from preprocess import preprocess_text
from remove_stopwords_and_normal import *
#from fandy_clean_text import clean_text

# example
text = r"What is the capital of Italy"

print("preprocess text: ", preprocess_text(text))

print(type(preprocess_text(text)))