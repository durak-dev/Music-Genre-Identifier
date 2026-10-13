import re
import nltk
import contractions
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Keep these global so they aren't rebuilt for every single word/row
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.PorterStemmer()
DETOKENIZER = TreebankWordDetokenizer()

STOPWORDS = set(nltk.corpus.stopwords.words("english") + [
    "oh", "yeah", "la", "na", "da", "ay", "ah", "woah", "woo", "uh", "ha", "hey"
])


def regex_cleaner(raw_text):
    sections = r"verse|chorus|intro|outro|bridge|hook|refrain|pre-chorus"
    song_tags = re.compile(rf"[\[\(]?\b(?:{sections})\b(?:\s+\d+|\s*:)?\s*[\]\)]?", flags=re.IGNORECASE)
    url_pattern = r"(?:https?://|www\.)\S+|(?:[a-z0-9]+\.)+[a-z0-9]{2,}(?:/\S*)?"
    bracket_pattern = r"\[[^\]]*\]|\([^)]*\)"
    hashtags_at_pattern = r"[#@\u0040\uFF20\uFE6B]"
    newline_pattern = r"\r?\n+"
    punctuation_pattern = r"[\u0021-\u0026\u0028-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff]+"
    multi_space_pattern = r"\s+"

    clean_text = re.sub(song_tags, "", raw_text)
    clean_text = re.sub(url_pattern, "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(bracket_pattern, "", clean_text)
    clean_text = re.sub(hashtags_at_pattern, "", clean_text)
    clean_text = re.sub(newline_pattern, " ", clean_text)
    clean_text = re.sub(punctuation_pattern, "", clean_text)
    clean_text = re.sub(multi_space_pattern, " ", clean_text).strip()
    return clean_text


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