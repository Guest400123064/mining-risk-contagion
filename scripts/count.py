# %%
from typing import List, Set, Dict

import json

import tqdm
import logging

logging.basicConfig(filename='log/count-agree-words.log', 
                    filemode='w',
                    level=logging.WARNING)

from collections import Counter

from transformers import pipeline

import re
import string
import unicodedata

from nltk.corpus import wordnet2021 as wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd


class SynonymGenerator:

    def __init__(self):

        self.thresh = 0.3
        self.model_nli = pipeline(
            task='text-classification',
            model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
            device='cuda:0',
            top_k=3
        )

    def generate(self, seeds: List[str], num_iter: int = 1) -> List[str]:
        """"""

        # Type checking
        if isinstance(seeds, str):
            seeds = [seeds]
        elif not isinstance(seeds, list):
            raise TypeError(f'< seeds > should be either `List[str]` or `str`, found `{type(seeds)}` instead')

        # Iteratively augment seeds
        ret = set(seeds)
        for _ in range(num_iter):
            for s in ret:
                ret = ret.union(self._generate_single(s))
        return list(ret)

    def _generate_single(self, seed: str) -> Set[str]:
        
        # Augmentation
        candidates = set()
        for synset in wordnet.synsets(seed.lower()):
            for lemma_name in synset.lemma_names():
                candidates.add(lemma_name.lower().replace('_', ' '))
        
        if not candidates:
            return set()

        # Filter with NLI classifier
        ret = set()
        candidates = list(candidates)
        infers = self.model_nli([f'{seed}. {c}' for c in candidates])
        for c, scores in zip(candidates, infers):
            for pred in scores:
                if pred['label'] == 'entailment' and pred['score'] > self.thresh:
                    ret.add(c)
        return ret


class WordCounter:

    def __init__(self, target_words: List[str]):

        self.wnl = WordNetLemmatizer()
        self.tgt = [(w, self.wnl.lemmatize(w, pos='v')) for w in target_words]
        self.stp = set(stopwords.words('english'))

    def preprocess(self, s: str) -> List[str]:

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
                            r' ')
        
        # Check if processed string is null
        s = re.sub(r'[^' + pattern_keep + r']+', '', s)
        if s == '':
            logging.warning(f'@ {self.__class__.__name__}.preprocess() :: ' + 
                f'Null string after preprocessing; input: <{i}>')
            return ['[NULL]']

        # Tokenize, remove stopwords, and lemmatize each word to verb
        ret = [self.wnl.lemmatize(w, pos='v') for w in word_tokenize(s.lower()) 
                    if w.lower() not in self.stp]
        return ret

    def count(self, text: str) -> Dict[str, int]:

        processed = self.preprocess(text)
        counter = Counter(processed)
        ret = {'num_risk_factors_tokens': len(processed),
               'num_conciliatory_tokens': 0}

        for word, lemma in self.tgt:
            ret[word] = counter.get(lemma, 0)
            ret['num_conciliatory_tokens'] += counter.get(lemma, 0)
        return ret


# %%
if __name__ == '__main__':

    # Load agreement words
    with open('models/agreement-words.json') as f:
        counter = WordCounter(json.load(f).get('augmentation'))

    # Load pre-extracted item-1a files
    with open('data/extracts/item1a-full.json') as f:
        item1a = json.load(f)

    # Count
    counts = []
    for f in tqdm.tqdm(item1a):
        result = {'symbol': f['symbol'], 'filing_time': f['filing_time']}
        result.update(counter.count(f['item1a']))
        counts.append(result)

    # To csv
    pd.DataFrame(counts).to_csv('data/statistics/count-agree-words.csv', index=False)
