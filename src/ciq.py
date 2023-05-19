# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-02-2023
# =============================================================================

from typing import Callable, List, Union, Generator
from collections import namedtuple

import warnings

import wrds
import sqlalchemy

import pandas as pd


SHARED_USERNAME = "annejam"


def get_wrds_connection(username: str = SHARED_USERNAME) -> wrds.Connection:
    """Get a connection to WRDS database. Requires a username, 
        and it will prompt you to input a password."""
        
    return wrds.Connection(wrds_username=username)


def get_ciq_id(conn: wrds.Connection, 
               company_name: str, 
               country_id: int,
               allow_no_country: bool = True) -> int:
    """Get the CIQ id for a company, given its name and country."""

    company_name = company_name.lower()
    
    sql = sqlalchemy.text(f"""
        SELECT  companyId AS company_id
        FROM    ciq.ciqCompany
        WHERE   LOWER(companyName) LIKE '{company_name}%'
                AND countryId = {country_id}
        LIMIT   1;
    """)
    df = conn.raw_sql(sql)

    if df.empty and allow_no_country:
        warnings.warn(f"Company {company_name} not found in country {country_id}; "
                        "trying without country restriction. Set `allow_no_country` "
                        "to `False` to disable this feature.")
        sql = sqlalchemy.text(f"""
            SELECT      companyId AS company_id
            FROM        ciq.ciqCompany
            WHERE       LOWER(companyName) LIKE '{company_name}%'
            ORDER BY    countryId DESC  -- For reproducibility only
            LIMIT       1;
        """)
        df = conn.raw_sql(sql)

    # If still empty, raise an error
    if df.empty:
        raise ValueError(f"Company {company_name} not found in country {country_id}.")
    return df.company_id[0]


def get_earnings_call_transcript_detail(conn: wrds.Connection,
                                        company_id: int) -> pd.DataFrame:
    """Get the quarterly earning's call transcript detail for a company, 
        given its CIQ id. It will only retrieve the latest audited version 
        as they are the most reliable."""

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
            detail.companyId                            AS ciq_id
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
    df = conn.raw_sql(sql).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Company {company_id} transcript not found.")
    return df


def make_country2id_fn(conn: wrds.Connection) -> Callable[[str], int]:
    """Make a function that takes a country name and returns 
        the corresponding country id."""

    sql = sqlalchemy.text("""
        SELECT  countryId AS country_id,
                country AS country_name 
        FROM    ciq.ciqCountryGeo;
    """)

    df = conn.raw_sql(sql)
    country2id = {country_name.lower(): country_id for country_id, country_name in df.values}
    return lambda country_name: country2id[country_name.lower()]
