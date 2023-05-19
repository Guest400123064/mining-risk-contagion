from typing import Type, Sequence, Union

import datetime

import os
import pathlib
import glob
import shutil

import json
import pandas as pd

import logging
import warnings

import secedgar as sec
from secedgar.parser import MetaParser
from secedgar.exceptions import EDGARQueryError, NoFilingsError


class Fetcher:

    def __init__(self, save_dir: str, user_agent: str, filing_type: Type[sec.FilingType]) -> None:

        self.save_dir    = save_dir
        self.user_agent  = user_agent
        self.filing_type = filing_type
        self.parser      = MetaParser()
        self.start_date  = datetime.date(2000, 12, 31)

    def process(self, ciks: Union[str, Sequence[str]]) -> Sequence[bool]:
        """Simple wrapper for batched processing of cik(s)"""

        if isinstance(ciks, str):
            ciks = [ciks]
        return [self._fetch_single(cik.lower().strip()) for cik in ciks]

    def _fetch_single(self, cik: str) -> bool:
        """Driver methods to retrieve filings for a single company (cik)"""

        # Remove out dir if exists
        dout = os.path.join(self.save_dir, cik, self.filing_type.value)
        if os.path.exists(dout):
            shutil.rmtree(dout)

        try:
            # Fetch from EDGAR db
            filings = sec.filings(cik, self.filing_type, self.user_agent, self.start_date)
            filings.save(self.save_dir)

            # Parse try parse meta-data and each section
            fpattern = os.path.join(dout, '*.txt')
            for fpath in glob.glob(fpattern):
                self._parse_single(fpath)
            return True

        # Error handling
        except EDGARQueryError as e:
            warnings.warn(f'Unknown CIK < {cik} >; skipped')
            return False
        except NoFilingsError as e:
            warnings.warn(f'No filing < {self.filing_type.value} > found for CIK < {cik} >; skipped')
            return False
        except Exception as e:
            warnings.warn(f'Unknown exception when retrieving < {cik} > filing < {self.filing_type.value} >\n' + 
                          f'Error message: \n\t{e}')
            return False

    def _parse_single(self, fpath: str):

        # Save file name for re-organization
        dout = os.path.splitext(fpath)[0]

        # Process files
        logging.info(f'Processing file @ < {fpath} >')
        self.parser.process(fpath)

        # Move source file to out-dir
        shutil.move(fpath, os.path.join(dout, '__RAW__.htm'))
        shutil.move(os.path.join(dout, '0.metadata.json'), os.path.join(dout, '__META__.json'))

        # Rename according to metadata
        with open(os.path.join(dout, '__META__.json')) as f:
            metadata = json.load(f)

        droot = pathlib.Path(dout).parent
        os.rename(dout, droot / metadata['FILED_AS_OF_DATE'])


if __name__ == '__main__':

    df_comp = pd.read_csv('largest_mining_companies_by_market_cap.csv')
    fetcher = Fetcher('data', 'wangy49@seas.upenn.edu', sec.FilingType.FILING_20F)
    fetcher.process(df_comp.symbol.values)
