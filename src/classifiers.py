# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-10-2023
# =============================================================================

from typing import Union, Callable, List

import numpy as np


def make_setfit_binary_predictor(model_path: str) -> Callable[[Union[str, List[str]]], List[int]]:
    """Make a predictor function that takes a list of texts and 
        returns a list of booleans indicating whether the text is 
        <SOME_ATTRIBUTE> or not. The input `model_path` is the path to the 
        trained SetFit model."""
    
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(model_path)

    def predict_fn(texts: Union[str, List[str]],
                   thresh: float = 0.5,
                   return_probas: bool = False) -> List[int]:
        """Predict whether the input sequence of texts is noise or not."""
        
        if isinstance(texts, str):
            texts = [texts]

        probas = model.predict_proba(texts, as_numpy=True)[:, -1]
        if return_probas:
            return probas.tolist()
        return (probas > thresh).astype(int).tolist()

    return predict_fn
