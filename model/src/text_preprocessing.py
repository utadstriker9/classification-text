import re
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string 
# import utils
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string
    
# Cleaning Data
def clean_data(data, CONFIG_DATA, return_file=True):
    # Drop NA
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].dropna()
    
    # Lowering Case
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].str.lower()
    
    # Remove Non ASCII
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].str.encode('ascii', 'ignore').str.decode('ascii')
    
    # Remove Whitespace in Start and End
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].str.strip()

    # Punctuation Removal Code
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].apply(lambda text: re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', str(text)))
    
    # Remove Mention, Link, Hashtag, etc
    def replace_str(text):
        cleaned_text = re.sub(r'[\n\r\t]', ' ', str(text)) # Remove Tab, Enter, Space
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text)) # Remove Many Hwer
        cleaned_text = re.sub(r"\b[a-zA-Z]\b", "", str(text)) # Remove 1 Char Only
        cleaned_text = re.sub(r'\d+', '', str(text)) # Remove Number
        return cleaned_text

    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].apply(replace_str)
    
    # Remove incomplete URL
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace("http://", " ").replace("https://", " ")
     
    # Remove Multiple Space
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace(r'\s+', ' ', regex=True)
    
    if return_file:
        return data

# Generate Preprocessor
def generate_preprocessor(data, CONFIG_DATA, return_file=False):
    # clean data
    data = clean_data(data, CONFIG_DATA)
    
    # tokenization 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    data[CONFIG_DATA['token_column']] = data[CONFIG_DATA['text_column']].apply(word_tokenize_wrapper)
    
    # Stopwords Removal (Filtering)
    list_stopwords = stopwords.words("indonesian")
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", "klo", 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])
    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    data[CONFIG_DATA['token_column']] = data[CONFIG_DATA['token_column']].apply(stopwords_removal)
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in data[CONFIG_DATA['token_column']]:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)

    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    data[CONFIG_DATA['token_column']] = data[CONFIG_DATA['token_column']].swifter.apply(get_stemmed_term)
    
    # Print Data
    print('ready processed, data shape   :', data.shape)
    
    if return_file:
        return data