import re
import nltk
import contractions
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Keep these global so they aren't rebuilt for every single word/row
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.PorterStemmer()
DETOKENIZER = TreebankWordDetokenizer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


import re
import nltk


def regex_cleaner(raw_text, stopwords=None, collapse_elongations=True):
    if stopwords is None:
        stopwords = STOPWORDS

    sections = r"pre-chorus|verse|chorus|intro|outro|bridge|hook|refrain"
    song_tags = re.compile(rf"[\[\(]?\b(?:{sections})\b(?:\s+\d+|\s*:)?\s*[\]\)]?", flags=re.IGNORECASE)
    url_pattern = re.compile(r"(?:https?://|www\.)\S+|(?:[a-z0-9]+\.)+[a-z0-9]{2,}(?:/\S*)?", flags=re.IGNORECASE)
    bracket_pattern = re.compile(r"\[[^\]]*\]|\([^)]*\)")
    hashtags_at_pattern = re.compile(r"[#@\uFF20\uFE6B]")
    newline_pattern = re.compile(r"\r?\n+")
    word_with_digit_pattern = re.compile(r"\S*\d\S*")
    punctuation_pattern = re.compile(r"[\u0021-\u0026\u0028-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff]+")
    elongation_pattern = re.compile(r"([^\W\d_])\1{2,}")
    multi_space_pattern = re.compile(r"\s+")
    apostrophe_pattern = re.compile(r"[`´‘’ʼ]")

    text = raw_text.lower()
    text = apostrophe_pattern.sub("'", text)  # curly quotes must stay apostrophes so contractions.fix can expand them
    text = url_pattern.sub(" ", text)
    text = song_tags.sub(" ", text)
    text = bracket_pattern.sub(" ", text)
    text = hashtags_at_pattern.sub(" ", text)
    text = newline_pattern.sub(" ", text)

    # Delete any chunk containing a digit: 2you2, rr2, rr-2, rr_2, 90s, 2024
    text = word_with_digit_pattern.sub(" ", text)

    text = punctuation_pattern.sub(" ", text)

    if collapse_elongations:
        text = elongation_pattern.sub(r"\1\1", text)

    text = multi_space_pattern.sub(" ", text).strip()

    return " ".join(w for w in text.split() if w not in stopwords)

def lemmatize_all(token, list_pos=["n", "v", "a", "r", "s"]):
    for arg_1 in list_pos:
        token = LEMMATIZER.lemmatize(token, arg_1)
    return token


def main_pipeline(raw_text, 
                  print_output=False, 
                  no_stopwords=True,
                  convert_diacritics=True, 
                  lowercase=True, 
                  lemmatized=True,
                  list_pos=["n", "v", "a", "r", "s"],
                  stemmed=False, 
                  pos_tags_list=False,
                  tokenized_output=False,
                  **kwargs):

    clean_text = regex_cleaner(raw_text, **kwargs)
    clean_text = contractions.fix(clean_text)
    clean_text = re.sub(r"['’]s\b", "", clean_text)
    tokenized_text = nltk.tokenize.word_tokenize(clean_text)
    tokenized_text = [token for token in tokenized_text if token.isalpha()]

    if no_stopwords:
        tokenized_text = [item for item in tokenized_text if item.lower() not in STOPWORDS]
    
    if convert_diacritics:
        tokenized_text = [unidecode(token) for token in tokenized_text]
    
    if lowercase:
        tokenized_text = [item.lower() for item in tokenized_text]

    if lemmatized:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]
    
    if stemmed:
        tokenized_text = [STEMMER.stem(token) for token in tokenized_text]
    
    # Final gate after all transforms: plain ASCII letters only, 2+ chars, no stopwords
    tokenized_text = [t for t in tokenized_text
                      if len(t) > 1 and t.isascii() and t.isalpha()
                      and not (no_stopwords and t in STOPWORDS)]

    if pos_tags_list:
        pos_tags = [pos[1] for pos in nltk.tag.pos_tag(tokenized_text)]

    if print_output:
        print(raw_text)
        print(tokenized_text)
    
    if pos_tags_list:
        return (tokenized_text, pos_tags)
    
    if tokenized_output:
        return tokenized_text
    
    return str(DETOKENIZER.detokenize(tokenized_text))