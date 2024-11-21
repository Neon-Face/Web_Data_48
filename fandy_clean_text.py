import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Markdown bold
    text = re.sub(r'\d+', '', text)

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text



