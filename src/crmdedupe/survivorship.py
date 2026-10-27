"""Survivorship: pick the record that survives a merge, then fill its gaps from the duplicates."""
import pandas as pd

from . import normalize as N

STATUS_RANK = {"Customer": 0, "Churned": 1, "Prospect": 2}


def pick_account_masters(accounts: pd.DataFrame, cluster: dict, opp_counts: pd.Series) -> pd.DataFrame:
    """Rules, in order: paying customer beats churned beats prospect → more opportunities → created first."""
    a = accounts.assign(cluster_id=accounts.account_id.map(cluster),
                        _status=accounts.status.map(STATUS_RANK).fillna(9),
                        _opps=-accounts.account_id.map(opp_counts).fillna(0),
                        _created=pd.to_datetime(accounts.created_at))
    a = a.sort_values(["cluster_id", "_status", "_opps", "_created"])
    a["master_id"] = a.groupby("cluster_id").account_id.transform("first")
    return a


def golden_accounts(ranked: pd.DataFrame) -> pd.DataFrame:
    """One clean row per real company: master values, gaps filled from duplicates, formats standardized."""
    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    g = ranked.groupby("master_id", sort=False).agg(
        account_name=("account_name", "first"), website=("website", first_valid), industry=("industry", first_valid),
        employee_count=("employee_count", first_valid), segment=("segment", "first"),
        billing_state=("billing_state", first_valid), owner_id=("owner_id", "first"), csm_id=("csm_id", first_valid),
        status=("status", "first"), created_at=("created_at", "min"), merged_records=("account_id", "size"))
    g["website"] = g.website.map(lambda w: "www." + N.domain(w) if w else None)
    g["billing_state"] = g.billing_state.map(N.state)
    return g.reset_index().rename(columns={"master_id": "account_id"})


def pick_contact_masters(contacts: pd.DataFrame, cluster: dict) -> pd.DataFrame:
    """Rules: primary buyer → came from a lead → valid work email → created first."""
    c = contacts.assign(cluster_id=contacts.contact_id.map(cluster),
                        _primary=~contacts.is_primary.astype(bool),
                        _lead=contacts.lead_id.isna(),
                        _bad_email=contacts.email.map(lambda e: N.email(e) == "" or N.email_domain(e) in N.FREE_MAIL),
                        _created=pd.to_datetime(contacts.created_at))
    c = c.sort_values(["cluster_id", "_primary", "_lead", "_bad_email", "_created"])
    c["master_id"] = c.groupby("cluster_id").contact_id.transform("first")
    return c
