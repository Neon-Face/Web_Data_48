import spacy
import re
from num2words import num2words
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# Contraction patterns
CONTRACTION_MAP = {
    "can't": "can not",
    "won't": "will not",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "they're": "they are",
    "we're": "we are",
    "we've":"we have",
    "i've":"i have",
    "you've":"you have"
}

# Define words to keep
keep_words = {"not","cannot"}

# Expand contractions
def expand_contractions(text):
    for contraction, expanded in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)
    return text

# Convert numbers to words
def n2w(token_text):
    try:
        # Check if the text is an ordinal (e.g., "21st", "3rd")
        if re.match(r'^\d+(st|nd|rd|th)$', token_text):
            number = re.sub(r'(st|nd|rd|th)$', '', token_text)  # Remove the ordinal suffix
            return num2words(int(number), to='ordinal')
        # Handle regular numbers (integers or floats)
        return num2words(float(token_text))
    except (ValueError, TypeError):
        # Return the original text if conversion fails
        return token_text

# Spelling correction function
def correct_spelling(text):
    corrected_words = []
    doc = nlp(text)

    for token in doc:
        # Skip correction for named entities or proper nouns
        if token.ent_type_ or token.pos_ == "PROPN":
            corrected_words.append(token.text)
        else:
            # Check if word is in the dictionary, otherwise correct it
            corrected_word = spell.correction(token.text) 
            corrected_words.append(corrected_word if corrected_word else token.text)
    return " ".join(corrected_words)

# Main preprocessing function
def preprocess_text(text):
    # Transform AIGC "‘" and "’" 
    text = text.replace("‘", "'").replace("’", "'").replace("%"," percentage")

    # Lowercase the text
    text = text.lower()

    # Expand contractions
    text = expand_contractions(text)
    
    # Correct spelling
    text = correct_spelling(text)

    # Tokenize with spaCy
    doc = nlp(text)
    
    # Process tokens
    processed_tokens = []
    for token in doc:
        # Convert numbers to words
        token_text = n2w(token.text) if token.like_num else token.lemma_

        # Check if token should be included
        if (not token.is_stop or token.text.lower() in keep_words  # Remove stop words, but not remove words in keep word
            and not token.is_punct  # Remove punctuation
            and not token.is_space  # Remove spaces
            and len(token_text) >= 1):  # Remove short tokens

            processed_tokens.append(token_text)

    # Join tokens back into a single string
    processed_text = " ".join(processed_tokens)

    # Remove special characters (keep alphanumeric and spaces only)
    processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', processed_text)

    # Strip extra whitespace
    processed_text = processed_text.strip()

    return processed_text
