# %%
import tqdm
import logging

from collections import defaultdict

import json

import pandas as pd
import numpy as np

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


if __name__ == '__main__':

    k_keywords = 8
    k_entities = 8
    k_n_chunks = 8

    corpus_keywords = defaultdict(list)
    corpus_entities = defaultdict(list)
    corpus_n_chunks = defaultdict(list)

    with open('../data/extracts/item1a-full-scored.json', 'r') as f:
        texts = json.load(f)
        df = pd.DataFrame(texts).drop("status", axis=1)
        for text in texts:
            symbol = text['symbol']
            corpus_keywords[symbol].extend(text['keywords'])
            corpus_entities[symbol].extend(text['entities'])
            corpus_n_chunks[symbol].extend(text['noun_chunks'])

    texts = [{"symbol": symbol} for symbol in corpus_keywords.keys()]
    corpus_keywords = [list(set(v)) for v in corpus_keywords.values()]
    corpus_entities = [list(set(v)) for v in corpus_entities.values()]
    corpus_n_chunks = [list(set(v)) for v in corpus_n_chunks.values()]
        
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
        text[f'top{k_keywords}_key_scr_avg'] = np.nanmean(score[index])
        text[f'top{k_keywords}_key_scr_min'] = np.nanmin(score[index])
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
        text[f'top{k_entities}_ent_scr_avg'] = np.nanmean(score[index])
        text[f'top{k_entities}_ent_scr_min'] = np.nanmin(score[index])
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
        text[f'top{k_n_chunks}_nck_scr_avg'] = np.nanmean(score[index])
        text[f'top{k_n_chunks}_nck_scr_min'] = np.nanmin(score[index])
    logging.info('Done!')
    
    # Global ranking
    cols_score = ["top8_key_scr_avg", "top8_ent_scr_avg", "top8_nck_scr_avg", "top8_combo_avg",
                  "top8_key_scr_min", "top8_ent_scr_min", "top8_nck_scr_min", "top8_combo_min"]
    
    df_global = pd.DataFrame(texts)
    df_global.loc[:, "top8_key"] = df_global.apply(lambda r: json.dumps(dict(zip(r["top8_key"], r["top8_key_scr_all"])), indent=True), axis=1)
    df_global.drop("top8_key_scr_all", axis=1, inplace=True)
    
    df_global.loc[:, "top8_ent"] = df_global.apply(lambda r: json.dumps(dict(zip(r["top8_ent"], r["top8_ent_scr_all"])), indent=True), axis=1)
    df_global.drop("top8_ent_scr_all", axis=1, inplace=True)
    
    df_global.loc[:, "top8_nck"] = df_global.apply(lambda r: json.dumps(dict(zip(r["top8_nck"], r["top8_nck_scr_all"])), indent=True), axis=1)
    df_global.drop("top8_nck_scr_all", axis=1, inplace=True)
    
    df_global.loc[:, "top8_combo_avg"] = (df_global.loc[:, "top8_key_scr_avg"] + 
                                          df_global.loc[:, "top8_ent_scr_avg"] + 
                                          df_global.loc[:, "top8_nck_scr_avg"]) / 3.0
    df_global.loc[:, "top8_combo_min"] = (df_global.loc[:, "top8_key_scr_min"] + 
                                          df_global.loc[:, "top8_ent_scr_min"] + 
                                          df_global.loc[:, "top8_nck_scr_min"]) / 3.0
    
    for col_score in cols_score:
        df_global.loc[:, f"rank_global_{col_score}"] = df_global.loc[:, col_score].rank(method="min")
    
    # %%
    # Compute rankings
    df.loc[:, "keywords"] = df.loc[:, "keywords"].map(lambda ks: ",\n".join(ks))
    df.loc[:, "entities"] = df.loc[:, "entities"].map(lambda es: ",\n".join(es))
    df.loc[:, "noun_chunks"] = df.loc[:, "noun_chunks"].map(lambda ncs: ",\n".join(ncs))

    df.loc[:, "item1a"] = df.loc[:, "item1a"].map(lambda ps: "\n\n".join(ps))
    
    df.loc[:, "top8_key"] = df.apply(lambda r: json.dumps(dict(zip(r["top8_key"], r["top8_key_scr_all"])), indent=True), axis=1)
    df = df.drop("top8_key_scr_all", axis=1)
    
    df.loc[:, "top8_ent"] = df.apply(lambda r: json.dumps(dict(zip(r["top8_ent"], r["top8_ent_scr_all"])), indent=True), axis=1)
    df = df.drop("top8_ent_scr_all", axis=1)
    
    df.loc[:, "top8_nck"] = df.apply(lambda r: json.dumps(dict(zip(r["top8_nck"], r["top8_nck_scr_all"])), indent=True), axis=1)
    df = df.drop("top8_nck_scr_all", axis=1)
    
    # Convert scores to float
    cast_dict = {col: 'float' for col in df.filter(regex=r"scr", axis=1).columns}
    df = df.astype(cast_dict)

    # Combo scores    
    df.loc[:, "top8_combo_avg"] = (df.loc[:, "top8_key_scr_avg"] + 
                                   df.loc[:, "top8_ent_scr_avg"] + 
                                   df.loc[:, "top8_nck_scr_avg"]) / 3.0
    df.loc[:, "top8_combo_min"] = (df.loc[:, "top8_key_scr_min"] + 
                                   df.loc[:, "top8_ent_scr_min"] + 
                                   df.loc[:, "top8_nck_scr_min"]) / 3.0
    
    # Group by year rankings
    sr_years = df.loc[:, "filing_time"].map(lambda t: t.split("-")[0])
    for col_score in cols_score:
        df.loc[:, f"rank_by_year_{col_score}"] = df.groupby(sr_years)[col_score].rank(method="min")
    
    # Group by firm rankings
    for col_score in cols_score:
        df.loc[:, f"rank_by_firm_{col_score}"] = df.groupby("symbol")[col_score].rank(method="min")
    
    # Excel works but CSV fails with `\n`
    df.to_excel('uniqueness.xlsx', index=False)
    df_global.to_excel('uniqueness-global.xlsx', index=False)

# %%
