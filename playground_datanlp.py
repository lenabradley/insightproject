import data
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import re


def clean_text(par):
    ''' Given string, clean & standardize it (lowercase, a-zA-Z only)
    '''
    par = re.sub(r"[^a-z]+", " ", par.lower()).strip()
    return par

def split_tokenize_criteria(par, group='in'):
    ''' Given criteria text, extract and return in/exclusion criteria

    Input:
        par (str): String with criteria text to process
        group (int): If 'in' ('ex') return inclusion (exclusion) tokens

    Return:
        substr (str): inclusion or exclusion criteria text
    '''

    # Where do in/exclusion keywords start/stop?
    in_keyword = 'inclusion criteria'
    ex_keyword = 'exclusion criteria'
    in_start, ex_start = par.find(in_keyword), par.find(ex_keyword)
    in_end = -1
    if in_start >= 0:
        in_end = in_start + len(in_keyword)
    ex_end = -1
    if ex_start >= 0:
        ex_end = ex_start + len(ex_keyword)

    # Based on keyword existence and order, extract text between keywords
    incl_text, excl_text = '', ''

    # Both keywords found
    if in_start >= 0 and ex_start >= 0:

        # inclusion before exclusion
        if ex_start > in_start:
            incl_text = par[in_end:ex_start]
            excl_text = par[ex_end:]

        # exclusion before inclusion
        elif in_start < ex_start:
            incl_text = par[in_end:]
            excl_text = par[ex_end:in_start]

    # Only inclusion found
    elif in_start >= 0:
        incl_text = par[in_end:]

    # Only exclusion found
    elif ex_start >= 0:
        excl_text = par[ex_end:]

    # Return requested substring
    substr = []
    if group == 'in':
        substr = incl_text.strip()
    elif group == 'ex':
        substr = excl_text.strip()

    return substr


# =================================================


# Connect to database
engine = data._connectdb()
df = pd.read_sql_table('eligibilities', engine)
df = df[['nct_id', 'criteria']].head(100)


# Clean text
df['clean_criteria'] = df['criteria'].apply(clean_text)

# Extract in/exclusion criteria
df["incl"] = df["clean_criteria"].apply(split_tokenize_criteria, group='in')
df["excl"] = df["clean_criteria"].apply(split_tokenize_criteria, group='ex')


# Tokenize criteria & gather vocabulary
tokenizer = RegexpTokenizer(r'\w+')
df["incl_tokens"] = df["incl"].apply(tokenizer.tokenize)
df["excl_tokens"] = df["excl"].apply(tokenizer.tokenize)
all_words = ([word for tokens in df["incl_tokens"] for word in tokens] + 
    [word for tokens in df["excl_tokens"] for word in tokens])
vocab = set(all_words)

#  Bag of words counts
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer


def cv(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

X = df["incl"].tolist()
X_counts, count_vectorizer = cv(X)

vectorizer = CountVectorizer()
