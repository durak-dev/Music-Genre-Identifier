import re
import nltk
import contractions
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Keep these global so they aren't rebuilt for every single word/row
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.PorterStemmer()
DETOKENIZER = TreebankWordDetokenizer()
# Negations are kept: removing them flips the meaning of a line ("I do not love you" -> "love")
NEGATIONS = {"no", "nor", "not"}
STOPWORDS = set(nltk.corpus.stopwords.words("english")) - NEGATIONS


import re
import nltk


def regex_cleaner(raw_text, collapse_elongations=True):
    # Stopwords are no longer removed here: this runs before contractions are expanded,
    # so "don't" would survive and only become a stopword later. main_pipeline removes them once.
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

    text = raw_text.lower()
    # Curly apostrophes fall inside the punctuation range below, which split "don’t" into "don t"
    # before contractions.fix could expand it; normalise them to a straight apostrophe first
    text = text.replace("\u2019", "'").replace("\u2018", "'")
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

    return text

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