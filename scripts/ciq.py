from typing import Callable, NoReturn, List, Union, Generator
from collections import namedtuple

import os
import pathlib

import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import tqdm
import logging

import wrds
import sqlalchemy

import pandas as pd

from src.misc import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

CompanyInfo = namedtuple("CompanyInfo", ["company_name", "country"])
ExecResult  = namedtuple("ExecResult",  ["company_name", "country", "is_success", "error_message"])


class Downloader:

    SHARED_USERNAME = "annejam"

    def __init__(self, data_folder: Union[str, pathlib.Path]) -> NoReturn:
        
        self.db = wrds.Connection(wrds_username=self.SHARED_USERNAME)
        self.data_folder = pathlib.Path(data_folder)
        self.country2id = self._make_country2id()

    def __call__(self, companies: List[CompanyInfo]) -> Generator[ExecResult, None, None]:
        
        for company in tqdm.tqdm(companies):
            try:
                company_id = self.get_company_id(company.company_name, company.country)
                df = self.get_transcript_detail(company_id)
                df.to_json(self.data_folder / f"{company.company_name}.json",
                           orient="records",
                           indent=4)
                yield ExecResult(company.company_name, company.country, True, "")
            except Exception as e:
                yield ExecResult(company.company_name, company.country, False, str(e))

    def _make_country2id(self) -> Callable[[str], int]:

        sql = sqlalchemy.text("""
            SELECT  countryid AS country_id,
                    country AS country_name 
            FROM    ciq.ciqcountrygeo;
        """)

        df = self.db.raw_sql(sql)
        tb = {c.lower(): int(i) for i, c in 
                zip(df.country_id, df.country_name)}
        return lambda c: tb[c.lower()]

    def get_company_id(self, company_name: str, country: str) -> int:

        country_id = self.country2id(country)
        company_name = company_name.strip().lower()

        sql = sqlalchemy.text(f"""
            SELECT      companyid AS company_id
            FROM        ciq.ciqcompany
            WHERE       LOWER(companyname) LIKE '{company_name}%'
                        AND countryid = {country_id}
            LIMIT       1;
        """)
        df = self.db.raw_sql(sql)

        # Try no country restriction
        if df.empty:
            logging.info(f"Company {company_name} not found in {country}; "
                            "retrying without country restriction.")
            sql = sqlalchemy.text(f"""
                SELECT      companyid AS company_id
                FROM        ciq.ciqcompany
                WHERE       LOWER(companyname) LIKE '{company_name}%'
                ORDER BY    countryid DESC
                LIMIT       1;
            """)
            df = self.db.raw_sql(sql)

        # Raise exception if still no company found
        if df.empty:
            raise Exception(f"Company Id {company_name} not found.")
        
        logging.info(f"Company {company_name} found with id {int(df.company_id[0])}.")
        return df.company_id[0]
    
    def get_transcript_detail(self, company_id: int) -> pd.DataFrame:

        sql = sqlalchemy.text(f"""
            WITH detail AS (
                SELECT
                    companyId
                    , companyName
                    , headline
                    , mostImportantDateUTC
                    , keyDevId
                    , transcriptId
                    , transcriptCollectionTypeName
                    , transcriptPresentationTypeName
                FROM (
                    SELECT 
                        companyId
                        , companyName
                        , headline
                        , mostImportantDateUTC
                        , keyDevId
                        , transcriptId
                        , transcriptCollectionTypeName
                        , transcriptPresentationTypeName
                        , RANK() OVER (PARTITION BY keyDevId ORDER BY 
                                transcriptCreationDate_UTC DESC, 
                                transcriptCreationTime_UTC DESC) AS rank
                    FROM
                        ciq_transcripts.wrds_transcript_detail
                    WHERE
                        companyId = {company_id}
                        AND keyDevEventTypeId = 48  -- Code 48 is Earnings Call
                        AND mostImportantDateUTC >= '2008-01-01'
                ) AS ordered_by_creation_time
                WHERE rank = 1
            )
            
            SELECT
                detail.companyId                            AS company_id
                , detail.companyName                        AS company_name
                , detail.headline                           AS headline
                , detail.mostImportantDateUTC               AS most_important_date_utc
                , detail.keyDevId                           AS key_dev_id                  
                , detail.transcriptId                       AS transcript_id
                , detail.transcriptCollectionTypeName       AS transcript_collection_type_name
                , detail.transcriptPresentationTypeName     AS transcript_presentation_type_name
                , person.transcriptComponentId              AS transcript_component_id
                , person.componentOrder                     AS component_order
                , person.transcriptComponentTypeName        AS transcript_component_type_name
                , person.transcriptPersonId                 AS transcript_person_id
                , person.transcriptPersonName               AS transcript_person_name
                , person.speakerTypeName                    AS speaker_type_name
                , component.componentText                   AS component_text
            FROM 
                detail
                INNER JOIN ciq_transcripts.wrds_transcript_person AS person
                    ON detail.transcriptId = person.transcriptId
                INNER JOIN ciq_transcripts.ciqTranscriptComponent AS component
                    ON person.transcriptComponentId = component.transcriptComponentId
            ORDER BY
                key_dev_id
                , transcript_id
                , component_order;
        """)
        df = self.db.raw_sql(sql).reset_index(drop=True)

        if df.empty:
            raise Exception(f"Company {company_id} has no transcript.")
        return df
    
    def close(self):
        self.db.close()


if __name__ == "__main__":

    home_folder = pathlib.Path(__file__).parent.parent
    meta_folder = home_folder / "data" / "meta"
    tran_folder = home_folder / "data" / "transcripts"

    df_companies = pd.read_csv(meta_folder / "largest_mining_companies_by_market_cap_with_filing_ciq_id.csv")
    companies = [CompanyInfo(n, c) for n, c in zip(df_companies.company_name, df_companies.country)]

    downloader = Downloader(tran_folder)
    has_transcript = []
    for result in downloader(companies):
        logging.info(result)
        has_transcript.append(result.is_success)
    
    df_companies["has_transcript"] = has_transcript
    df_companies.to_csv(meta_folder / "largest_mining_companies_by_market_cap_with_filing_ciq_id_with_transcript.csv", index=False)
    downloader.close()
