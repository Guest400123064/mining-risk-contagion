# %%
from typing import List, Tuple

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import sys
sys.path.append('..')

import json
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from functools import reduce
from operator import add

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
import keyspacy

from thinc.backends import set_gpu_allocator
from thinc.util import require_gpu

set_gpu_allocator('pytorch')
require_gpu(0)


class Analyzer:
    """Custom analyzer for CountVectorizer, which constraints the vocabulary to 
        key verbs, named entities, and noun chunks."""
    
    def __init__(self, ):
        
        self.max_chars = 1000000 * 2
        
        self.nlp = self.get_pipeline()
        self.wnl = WordNetLemmatizer()
        self.ent = {'NORP', 'ORG', 'GPE', 'LOC', 'EVENT', 'LAW'}
        
    def __call__(self, text: str) -> List[str]:
        
        text = text[:self.max_chars - 1]
        
        def _clean_span(s) -> str:
            ret = "".join(t.text_with_ws for t in s if not t.is_stop)
            ret = " ".join(self.wnl.lemmatize(t) for t in ret.lower().split())
            return ret.strip()

        keys, ents, ncks = set(), set(), set()
        doc = self.nlp(text)
        
        # Key verbs
        for sent in doc.sents:
            root = sent.root
            if not root.is_stop:
                keys.add(sent.root.lemma_.lower())

        # # Named entities
        # for ent in doc.ents:
        #     if ent.label_ in self.ent:
        #         ents.add(_clean_span(ent))

        # Noun chunks
        for nc in doc.noun_chunks:
            ncks.add(_clean_span(nc))
        
        return [t for t in (keys | ents | ncks) if not t.isnumeric()]

    def get_pipeline(self):
        """Load the pipeline for keywords extractor, which will be 
            called by the analyzer."""
    
        trf_config = {
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
        
        # Disable NER
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
        nlp.max_length = self.max_chars
        
        # Initialize transformer-based noun chunk extractor
        # nlp.add_pipe("transformer", config=trf_config)
        # nlp.add_pipe(keyspacy.component_name)
        # nlp.get_pipe('transformer').model.initialize()

        return nlp


# %%
if __name__ == "__main__":

    with open("../data/extracts/item1a-full-scored.json") as f:
        df = pd.DataFrame(json.load(f)).drop("status", axis=1)
        docs = list(set([p for p in reduce(add, df.item1a.values.tolist(), []) if len(p.split()) <= 128]))
    
    seed = [["human resources", "personnel", "attract", "management", "retain", "skilled", "executive"],
            ["intellectual property", "protection", "patent"],
            ["control", "law", "regulation"],
            ["catastrophes input prices", "operation"],
            ["volatile stock", "fluctuation", "revenue decline"],
            ["Shareholder's interest", "director", "officer"],
            ["financial condition", "macroeconomic cyclical industry"],
            ["competition"],
            ["debt", "credit"],
            ["capital", "funding"],
            ["financial condition"],
            ["property", "reserve", "oil", "gas"],
            ["tax", "federal"],
            ["foreign", "international", "currency"],
            ["accounting"],
            ["acquisition restructuring"],
            ["infrastructure"]]
    
    embed_model = SentenceTransformer("all-MiniLM-L12-v2").to("cuda:0")
    count_model = CountVectorizer(analyzer="word", 
                                  stop_words="english", 
                                  min_df=0.01)
    topic_model = BERTopic(min_topic_size=64, 
                           nr_topics="auto", 
                           top_n_words=20,
                           diversity=0.7, 
                           vectorizer_model=count_model,
                           embedding_model=embed_model,
                           verbose=True)

    topics, probs = topic_model.fit_transform(docs)
    topic_model.get_topic_info()

# %%
