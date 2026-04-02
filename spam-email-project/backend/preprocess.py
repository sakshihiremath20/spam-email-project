import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    stemmed = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed)

def preprocess_steps(text):
    original = text

    lowered = text.lower()
    cleaned_chars = re.sub(r'[^a-z0-9\s]', '', lowered)

    tokens = word_tokenize(cleaned_chars)
    all_tokens = tokens[:]

    filtered = [t for t in tokens if t not in stop_words and len(t) > 1]

    stemmed = [stemmer.stem(t) for t in filtered]

    return {
        'original_text': original,
        'lowercased': lowered,
        'tokens': all_tokens,
        'cleaned_text': filtered,
        'stemmed_text': stemmed,
        'final_text': ' '.join(stemmed)
    }