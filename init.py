import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')