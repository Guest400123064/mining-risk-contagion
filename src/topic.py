# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-21-2023
# =============================================================================

from typing import List, Dict, Union, NoReturn, Callable, Optional

import os
import pathlib

import numpy as np
import pandas as pd

from top2vec import Top2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text import tokenize_text


def generate_wordcloud(freq: Dict[str, int], topic_name: str = "Word Cloud"):
    """Generate a word cloud from a dictionary of word frequencies."""

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    fig, ax = plt.subplots(figsize=(16, 4), dpi=200)
    wc = WordCloud(width=1600, height=400).generate_from_frequencies(freq)
    
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(topic_name, loc="left", fontsize=25, pad=20)

    fig.tight_layout()
    ax.axis("off")
    return fig, ax


class Top2VecViewer:
    """Top2VecViewer is a wrapper class for Top2Vec model that provides shortcut properties 
        and methods for downstream tasks such as topic tagging, topic visualization, etc."""

    @classmethod
    def load(cls, model_path: str) -> "Top2VecViewer":
        return cls(Top2Vec.load(model_path))

    def __init__(self, model: Top2Vec) -> NoReturn:
        self.model = model
        self.null_topic_id = -1

    def get_topic_words(self, 
                        topic_nums: Optional[Union[int, List[int]]] = None,
                        is_reduced: bool = True, 
                        top_k: Optional[int] = None) -> pd.DataFrame:
        """Get the keywords of each topic."""

        if isinstance(topic_nums, int):
            topic_nums = [topic_nums]

        if topic_nums is not None:
            topic_nums = set(topic_nums)

        ret = []
        for words, scores, num in zip(*self.model.get_topics(reduced=is_reduced)):
            top_k = len(words) if top_k is None else top_k
            if topic_nums is None or num in topic_nums:
                obs = [(num, word, round(score, 4)) for k, (word, score) 
                        in enumerate(zip(words, scores)) if k < top_k]
                ret.extend(obs)
        return pd.DataFrame(ret, columns=["topic_num", "keyword", "cosine_score"])
    
    def make_t2v_predictor(self, 
                           is_reduced: bool = True) -> Callable[[Union[str, List[str]], float], List[List[int]]]:
        
        def predict_fn(texts: Union[str, List[str]],
                       thresh: float = 0.3,
                       return_all_scores: bool = False) -> Union[List[List[int]], np.ndarray]:
            """"""

            if isinstance(texts, str):
                texts = [texts]

            texts_tag = []
            sim_score = []
            for text in texts:
                _, _, sim, tags = self.model.query_topics(text, 
                                                          num_topics=5, 
                                                          reduced=is_reduced,
                                                          tokenizer=tokenize_text)
                sim_score.append(sim.tolist())
                if sim.max() <= thresh:
                    texts_tag.append([self.null_topic_id])
                    continue

                idx = np.where(sim > thresh)[0]
                texts_tag.append(tags[idx].tolist())

            if return_all_scores:
                return np.array(sim_score)
            return texts_tag

        return predict_fn

    def make_tfidf_predictor(self, 
                             is_reduced: bool = True,
                             topic_bags: Optional[List[List[str]]] = None) -> Callable[[Union[str, List[str]], float], List[List[int]]]:
        """Make a predictor function that takes a list of texts and 
            returns a list of tags (indices of topics). The input 
            `topic_token_bags` is a list of lists of tokens, where each
            bag of tokens represents a topic, should be extracted from 
            the trained Top2Vec model."""

        if topic_bags is None:
            topic_bags, _, _ = self.model.get_topics(reduced=is_reduced)
        
        vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
        topics_emb = vectorizer.fit_transform([" ".join(bag) for bag in topic_bags])

        def predict_fn(texts: Union[str, List[str]], 
                       thresh: float = 0.3, 
                       return_all_scores: bool = False) -> Union[List[List[int]], np.ndarray]:
            """Tag the input sequence of texts with the topics having cosine similarity 
                (between tf-idf vectors) greater than `thresh`. If no topic has similarity 
                greater than `thresh`, tag the text with the topic with the highest cosine similarity."""

            if isinstance(texts, str):
                texts = [texts]

            texts_tag = []
            texts_emb = vectorizer.transform(texts)
            sim_score = cosine_similarity(texts_emb, topics_emb)

            if return_all_scores:
                return sim_score

            for sim in sim_score:
                tags = np.where(sim > thresh)[0].tolist() \
                        if sim.max() > thresh else [self.null_topic_id]
                texts_tag.append(tags)
            return texts_tag
        
        return predict_fn
    
    def make_sbert_predictor(self, 
                             is_reduced: bool = True,
                             topic_bags: Optional[List[List[str]]] = None,
                             model_bert: str = "all-MiniLM-L6-v2") -> Callable[[Union[str, List[str]], float], List[List[int]]]:
        """Make a predictor function that takes a list of texts and
            returns a list of tags (indices of topics). The input
            `topic_bags` is a list of lists of tokens, where each
            bag of tokens represents a topic. If `topic_bags` is None,
            the function will extract the topic bags from the trained
            Top2Vec model. The input `model_bert` is the name of the
            pretrained SentenceBERT model used to encode the topic bags and
            the input texts."""

        from sentence_transformers import SentenceTransformer

        if topic_bags is None:
            topic_bags, _, _ = self.model.get_topics(reduced=is_reduced)

        model_bert = SentenceTransformer(model_bert)

        topics_emb = model_bert.encode([" ".join(bag) for bag in topic_bags])

        def predict_fn(texts: Union[str, List[str]], 
                       thresh: float = 0.3, 
                       return_all_scores: bool = False) -> Union[List[List[int]], np.ndarray]:
            """Tag the input sequence of texts with the topics having cosine similarity 
                (between tf-idf vectors) greater than `thresh`. If no topic has similarity 
                greater than `thresh`, tag the text with the topic with the highest cosine similarity."""

            if isinstance(texts, str):
                texts = [texts]

            texts_tag = []
            texts_emb = model_bert.encode(texts)
            sim_score = cosine_similarity(texts_emb, topics_emb)

            if return_all_scores:
                return sim_score

            for sim in sim_score:
                tags = np.where(sim > thresh)[0].tolist() \
                        if sim.max() > thresh else [self.null_topic_id]
                texts_tag.append(tags)
            return texts_tag
        
        return predict_fn

    def generate_topic_wordclouds(self, topic_nums: Union[int, List[int]], is_reduced: bool = True):
        """Generate wordclouds for the given topics."""

        if isinstance(topic_nums, int):
            topic_nums = [topic_nums]
    
        for topic_num in topic_nums:
            self.model.generate_topic_wordcloud(topic_num, reduced=is_reduced)
