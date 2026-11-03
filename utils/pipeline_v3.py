"""Lyrics preprocessing pipeline, version 3.

Combines `utils/pipeline_v2.py` and `transformers/pipeline_transformers.py`:

From pipeline_transformers.py
    * Stopwords are removed once in `main_pipeline`, after contractions are expanded
      (not in `regex_cleaner`, where "don't" would still be intact).
    * Negations ("no", "nor", "not") are kept, because removing them flips the meaning of a line.

From pipeline_v2.py
    * A wider set of apostrophe look-alikes is normalised to "'" before punctuation is stripped.
    * A final gate after all transforms keeps only plain ASCII letter tokens with 2+ characters
      that are not stopwords (this also removes the trailing-space tokens `unidecode` produces
      for CJK characters, and stopwords that only appear after lemmatization).

The public API is the same as before: `main_pipeline(raw_text, tokenized_output=True)`.
"""
import re

import contractions
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from unidecode import unidecode

# Keep these global so they aren't rebuilt for every single word/row
LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.PorterStemmer()
DETOKENIZER = TreebankWordDetokenizer()
# Negations are kept: removing them flips the meaning of a line ("I do not love you" -> "love")
NEGATIONS = {"no", "nor", "not"}
STOPWORDS = set(nltk.corpus.stopwords.words("english")) - NEGATIONS

# Compiled once at import time instead of on every call
_SECTIONS = r"pre-chorus|verse|chorus|intro|outro|bridge|hook|refrain"
SONG_TAGS = re.compile(rf"[\[\(]?\b(?:{_SECTIONS})\b(?:\s+\d+|\s*:)?\s*[\]\)]?", flags=re.IGNORECASE)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+|(?:[a-z0-9]+\.)+[a-z0-9]{2,}(?:/\S*)?", flags=re.IGNORECASE)
BRACKET_PATTERN = re.compile(r"\[[^\]]*\]|\([^)]*\)")
HASHTAGS_AT_PATTERN = re.compile(r"[#@＠﹫]")
NEWLINE_PATTERN = re.compile(r"\r?\n+")
WORD_WITH_DIGIT_PATTERN = re.compile(r"\S*\d\S*")
PUNCTUATION_PATTERN = re.compile(r"[!-&(-/:-?[-_‐-﻿]+")
ELONGATION_PATTERN = re.compile(r"([^\W\d_])\1{2,}")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
APOSTROPHE_PATTERN = re.compile(r"[`´‘’ʼ]")


def regex_cleaner(raw_text, collapse_elongations=True):
    """Regex based cleaning. Stopwords are NOT removed here, `main_pipeline` does that once,
    after contractions have been expanded."""
    text = raw_text.lower()
    text = APOSTROPHE_PATTERN.sub("'", text)
    text = URL_PATTERN.sub(" ", text)
    text = SONG_TAGS.sub(" ", text)
    text = BRACKET_PATTERN.sub(" ", text)
    text = HASHTAGS_AT_PATTERN.sub(" ", text)
    text = NEWLINE_PATTERN.sub(" ", text)

    # Delete any chunk containing a digit: 2you2, rr2, rr-2, rr_2, 90s, 2024
    text = WORD_WITH_DIGIT_PATTERN.sub(" ", text)

    text = PUNCTUATION_PATTERN.sub(" ", text)

    if collapse_elongations:
        text = ELONGATION_PATTERN.sub(r"\1\1", text)

    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def lemmatize_all(token, list_pos=["n", "v", "a", "r", "s"]):
    for pos in list_pos:
        token = LEMMATIZER.lemmatize(token, pos)
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
                  stopwords=None,
                  **kwargs):
    """Clean and tokenize one lyric.

    `stopwords` optionally replaces the default stopword set (negations kept).
    Remaining keyword arguments are passed to `regex_cleaner`.
    """
    if stopwords is None:
        stopwords = STOPWORDS

    clean_text = regex_cleaner(raw_text, **kwargs)
    clean_text = contractions.fix(clean_text)
    clean_text = re.sub(r"['’]s\b", "", clean_text)
    tokenized_text = nltk.tokenize.word_tokenize(clean_text)
    tokenized_text = [token for token in tokenized_text if token.isalpha()]

    if no_stopwords:
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]

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
                      and not (no_stopwords and t in stopwords)]

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
