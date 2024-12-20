import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
model_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')