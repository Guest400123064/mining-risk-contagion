# %%
import logging

logging.basicConfig(filename='../log/indexing.log', 
                    filemode='w',
                    level=logging.INFO)

import sys

sys.path.append('..')

import json
from typing import List, Tuple

import numpy as np
import tqdm
from sklearn.feature_extraction.text import CountVectorizer

CNTR_KEYWORDS = CountVectorizer(tokenizer=lambda x: x, 
                                lowercase=False,
                                min_df=3,
                                binary=True)
CNTR_ENTITIES = CountVectorizer(tokenizer=lambda x: x, 
                                lowercase=False,
                                min_df=1,
                                binary=True)
CNTR_N_CHUNKS = CountVectorizer(tokenizer=lambda x: x, 
                                lowercase=False,
                                min_df=3,
                                binary=True)

import re
import string
import unicodedata

import spacy
import keyspacy

from thinc.backends import set_gpu_allocator
from thinc.util import require_gpu

set_gpu_allocator('pytorch')
require_gpu(0)

CONFIG = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v3",        
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "tokenizer_config": {"use_fast": True},
        "get_spans": {
            "@span_getters": "spacy-transformers.strided_spans.v1",
            "window": 256,
            "stride": 246
        }
    }
}

NLP = spacy.load('en_core_web_lg')
NLP.add_pipe("transformer", config=CONFIG)
NLP.add_pipe(keyspacy.component_name)

ENT = {'NORP', 'ORG', 'GPE', 'LOC', 'EVENT', 'LAW'}

# Initialize transformer
NLP.get_pipe('transformer').model.initialize()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

WNL = WordNetLemmatizer()
SEP = '[SEP]'
STP = set(stopwords.words('english'))


def preprocess(s: str) -> List[str]:
    """Remove special characters/strings and tokenize
        into paragraphs with heuristics (split by `.\n`)"""
    
    # Normalize special chars
    i = s
    s = (unicodedata.normalize('NFKD', s)
            .encode('ascii', 'ignore')
            .decode())

    # Remove irrelevant info
    s = re.sub(r'\S*@\S*\s?', '', s)     # Email
    s = re.sub(r'\S*https?:\S*', '', s)  # URL
    
    # Keep punctuation and words only
    pattern_keep = (string.punctuation + 
                    string.ascii_letters + 
                    string.digits + 
                    r'\s')

    # Check if processed string is null
    s = re.sub(r'[^' + pattern_keep + r']+', '', s)
    if s == '':
        logging.warning(f'@ preprocess() :: ' + 
            f'Null string after preprocessing; input: <{i}>')
        return ['[NULL]']

    # To paragraphs
    return [re.sub(r'\s+', r' ', p) + '.' for p in s.split('.\n')]


def extract(ps: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Extract key verbs and named entities from list of paragraphs."""
    
    def _clean_span(s) -> str:
        ret = "".join(t.text_with_ws for t in s if not t.is_stop)
        ret = " ".join(WNL.lemmatize(t) for t in ret.lower().split())
        return ret.strip()

    verbs, ents, n_chunks = set(), set(), set()
    for p in NLP.pipe(ps, batch_size=32):
        p_verbs, p_ents, p_n_chunks = set(), set(), set()

        # Key verbs
        for sent in p.sents:
            root = sent.root
            if not root.is_stop:
                p_verbs.add(sent.root.lemma_.lower())

        # Named entities
        for ent in p.ents:
            if ent.label_ in ENT:
                p_ents.add(_clean_span(ent))

        # Noun chunks
        for nc, _ in p._.extract_keywords(use_mmr=True, diversity=0.7)[0]:
            p_n_chunks.add(" ".join(WNL.lemmatize(t) for t in nc.lower().split()))

        verbs    = verbs.union(p_verbs)
        ents     = ents.union(p_ents)
        n_chunks = n_chunks.union(p_n_chunks)

    return list(verbs), list(ents), list(n_chunks)

# %%
if __name__ == '__main__':
    
    k_keywords = 8
    k_entities = 8
    k_n_chunks = 8
    corpus_keywords = []
    corpus_entities = []
    corpus_n_chunks = []

    # Load pre-extracted item-1a files and extract keywords
    logging.info('Loading raw Item 1A files...')
    with open('../data/extracts/item1a-full.json') as f:
        texts = json.load(f)
        logging.info('Done!')
        
        logging.info('Extracting keywords and entities...')
        for text in tqdm.tqdm(texts):
            try:
                text['item1a'] = preprocess(text['item1a'])
                text['keywords'], text['entities'], text['noun_chunks'] = extract(text['item1a'])
                corpus_keywords.append(text['keywords'])
                corpus_entities.append(text['entities'])
                corpus_n_chunks.append(text['noun_chunks'])
            except Exception as e:
                logging.error(f'Unexpected error encountered when processing <{text["symbol"]}-{text["filing_time"]}>: {e}')
                text['keywords'] = ['[ERROR]']
                text['entities'] = ['[ERROR]']
                text['noun_chunks'] = ['[ERROR]']
                corpus_keywords.append(['[ERROR]'])
                corpus_entities.append(['[ERROR]'])
                corpus_n_chunks.append(['[ERROR]'])
        logging.info('Done!')

    # Count document frequency
    logging.info('Counting keywords...')
    counts_keywords = CNTR_KEYWORDS.fit_transform(corpus_keywords).A
    counts_entities = CNTR_ENTITIES.fit_transform(corpus_entities).A
    counts_n_chunks = CNTR_N_CHUNKS.fit_transform(corpus_n_chunks).A
    doc_freq_keywords = counts_keywords.sum(axis=0)
    doc_freq_entities = counts_entities.sum(axis=0)
    doc_freq_n_chunks = counts_n_chunks.sum(axis=0)
    logging.info('Done!')

    # Calculate scores according to document frequency
    scores_keywords = ((counts_keywords > 0) * doc_freq_keywords).astype(float)
    scores_keywords[scores_keywords == 0] = len(texts) + 1  # Rank to the bottom
    indices_keywords = np.argpartition(scores_keywords, k_keywords)[:, :k_keywords]
    
    # Grading keywords based on scores
    logging.info('Calculating mean/max keywords scores...')
    keywords = CNTR_KEYWORDS.get_feature_names_out()
    for text, score, index in tqdm.tqdm(zip(texts, scores_keywords, indices_keywords)):
        text[f'top{k_keywords}_key']         = keywords[index].tolist()
        text[f'top{k_keywords}_key_scr_all'] = [f'{s:.2f}' for s in score[index]]
        text[f'top{k_keywords}_key_scr_avg'] = f'{np.nanmean(score[index]):.2f}'
        text[f'top{k_keywords}_key_scr_min'] = f'{np.nanmin(score[index]):.2f}'
    logging.info('Done!')

    # Calculate scores according to document frequency
    scores_entities = ((counts_entities > 0) * doc_freq_entities).astype(float)
    scores_entities[scores_entities == 0] = len(texts) + 1
    indices_entities = np.argpartition(scores_entities, k_entities)[:, :k_entities]
    
    # Grading entities based on scores
    logging.info('Calculating mean/max entities scores...')
    keywords = CNTR_ENTITIES.get_feature_names_out()
    for text, score, index in tqdm.tqdm(zip(texts, scores_entities, indices_entities)):
        text[f'top{k_entities}_ent']         = keywords[index].tolist()
        text[f'top{k_entities}_ent_scr_all'] = [f'{s:.2f}' for s in score[index]]
        text[f'top{k_entities}_ent_scr_avg'] = f'{np.nanmean(score[index]):.2f}'
        text[f'top{k_entities}_ent_scr_min'] = f'{np.nanmin(score[index]):.2f}'
    logging.info('Done!')
    
    # Calculate scores according to document frequency
    scores_n_chunks = ((counts_n_chunks > 0) * doc_freq_n_chunks).astype(float)
    scores_n_chunks[scores_n_chunks == 0] = len(texts) + 1
    indices_n_chunks = np.argpartition(scores_n_chunks, k_n_chunks)[:, :k_n_chunks]
    
    # Grading noun chunks based on scores
    logging.info('Calculating mean/max noun chunks scores...')
    keywords = CNTR_N_CHUNKS.get_feature_names_out()
    for text, score, index in tqdm.tqdm(zip(texts, scores_n_chunks, indices_n_chunks)):
        text[f'top{k_n_chunks}_nck']         = keywords[index].tolist()
        text[f'top{k_n_chunks}_nck_scr_all'] = [f'{s:.2f}' for s in score[index]]
        text[f'top{k_n_chunks}_nck_scr_avg'] = f'{np.nanmean(score[index]):.2f}'
        text[f'top{k_n_chunks}_nck_scr_min'] = f'{np.nanmin(score[index]):.2f}'
    logging.info('Done!')

    # Write to file
    logging.info('Writing to file...')
    with open('../data/extracts/item1a-full-scored.json', 'w') as f:
        json.dump(texts, f, indent=True)
    logging.info('Done!')

# %%
