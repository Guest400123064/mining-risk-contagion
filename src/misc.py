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

from easydict import EasyDict


paths = EasyDict({"home": pathlib.Path(__file__).parent.parent})
paths.data        = paths.home  / "data"
paths.model       = paths.home  / "models"
paths.t2v         = paths.model / "top2vec"
paths.sec         = paths.data  / "sec"
paths.meta        = paths.data  / "meta"
paths.stats       = paths.data  / "statistics"
paths.extracts    = paths.data  / "extracts"
paths.transcripts = paths.data  / "transcripts"


def utc_to_date(utc: int) -> str:
    """Convert UTC timestamp to date string."""

    from datetime import datetime

    return datetime.utcfromtimestamp(int(utc) / 1000).strftime(r"%Y-%m-%d")


class LoggingHandler(logging.Handler):
    """Simple logging handler that writes to tqdm.tqdm.write."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)