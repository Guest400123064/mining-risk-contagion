# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-21-2023
# =============================================================================

from typing import List, Dict, Tuple, Set, Union, Optional, NoReturn, Callable

import os
import pathlib

import logging
import tqdm

import re
import string
import unicodedata

import nltk
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
except ImportError:
    nltk.download("wordnet")
    nltk.download("stopwords")
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

from src.misc import paths, LoggingHandler

wnl = WordNetLemmatizer()
stopwords = set(stopwords.words('english')) | \
                {"may", "could", "result", "mine", "coal", "finance", 
                 "price", "risk", "u", "company", "result", "financial", 
                 "mineral", "adverse", "adversely", "affect", "effect", "include"}

fasttext_model = None


def lemmatize_single_token(token: str) -> str:
    """Lemmatize the given token. Try all possible 
        lemmatization (n, v, a, r, s) until the input changes."""

    for pos in ["n", "v", "a", "r", "s"]:
        new_token = wnl.lemmatize(token, pos=pos)
        if new_token != token:
            return new_token
    return token


def tokenize_text(text: str, is_lemma: bool = True) -> List[str]:
    """Tokenize the given text using gensim's 
        simple_preprocess. Punctuations and stopwords
        are removed, and the tokens are lemmatized."""
    
    from gensim.utils import simple_preprocess
    
    lemma_fn = lemmatize_single_token if is_lemma else lambda x: x
    tokens = []
    for token in simple_preprocess(text, min_len=3):
        token = lemma_fn(token)
        if token not in stopwords:
            tokens.append(lemma_fn(token))
    return tokens


def normalize_text(text: str) -> str:
    """Normalize scraped text. This includes:
        - Replace all non-ascii characters to closest ascii
        - Remove irrelevant info such as email and url
        - Keep only alphabets, numbers, spaces, and punctuations
        - Misc"""

    # Replace all non-ascii characters to closest ascii
    text = unicodedata.normalize("NFKD", text) \
                      .encode("ascii", "ignore") \
                      .decode("utf-8", "ignore")
    
    # Remove irrelevant info if any
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"\S*https?:\S*", "", text)

    # Keep only alphabets, numbers, spaces, and punctuations
    pattern = r"[^a-zA-Z0-9\s" + string.punctuation + r"]"
    text = re.sub(pattern, "", text)

    # Misc
    text = text.replace("’", "'")
    return text


def text_to_sentences(text: str, min_num_tokens: int = 5) -> List[str]:
    """Split the given text into sentences using nltk's sent_tokenize."""

    from nltk.tokenize import sent_tokenize

    ret = []
    for sent in sent_tokenize(text):
        if len(sent.split()) >= min_num_tokens:
            ret.append(sent)
    return ret


def get_similar_words_fasttext(seeds: List[str], 
                               top_k: int = 20,
                               thresh: float = 0.6) -> List[str]:
    """Get similar words using fasttext with at least `thresh` similarity."""

    from gensim.models import KeyedVectors

    global fasttext_model

    if fasttext_model is None:
        fasttext_path = paths.model / "fasttext" / "wiki-news-300d-1M.vec"
        fasttext_model = KeyedVectors.load_word2vec_format(str(fasttext_path), binary=False)

    ret = []
    seeds = [s.lower() for s in seeds if s in fasttext_model]
    for w, s in fasttext_model.most_similar(seeds, topn=top_k):
        if s >= thresh:
            ret.append(w.lower())
            continue
        break
    return ret
