"""Contact (person) matching. Runs after account resolution so contacts on duplicate accounts compare as
belonging to the same company."""
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from . import normalize as N
from .blocking import candidate_pairs
from .model import Comparison

COMPARISONS = [
    Comparison("email", ["missing", "different", "same local part", "exact"]),
    Comparison("phone", ["missing", "different", "same"]),
    Comparison("last_name", ["missing", "different", "similar (JW ≥ 0.9)", "exact"]),
    Comparison("first_name", ["missing", "different", "nickname / prefix", "exact"]),
    Comparison("company", ["missing", "different", "same company"]),
    Comparison("title", ["missing", "different", "same"]),
]
BLOCKING_KEYS = ["n_email", "n_phone", "k_company_last", "k_name"]


def prepare(contacts: pd.DataFrame, account_cluster: dict) -> pd.DataFrame:
    c = contacts.copy()
    c["n_email"] = c["email"].map(N.email)
    c["n_local"] = c["n_email"].str.split("@").str[0].fillna("")
    c["n_phone"] = c["phone"].map(N.phone)
    c["n_first"] = c["first_name"].map(N.first_name)
    c["n_last"] = c["last_name"].map(N.person_name)
    c["company"] = c["account_id"].map(account_cluster).fillna(c["account_id"])
    c["k_company_last"] = c["company"] + "|" + c["n_last"]
    c["k_name"] = c["n_first"].str[:1] + "|" + c["n_last"]
    return c


def _first(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if a == b:
        return 3
    if len(min(a, b, key=len)) >= 3 and (a.startswith(b) or b.startswith(a)):
        return 2
    return 1


def compare(c: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    l = c.set_index("contact_id").loc[pairs.id_l].reset_index(drop=True)
    r = c.set_index("contact_id").loc[pairs.id_r].reset_index(drop=True)
    has_e = (l.n_email != "") & (r.n_email != "")
    email = np.select([~has_e, l.n_email == r.n_email, l.n_local == r.n_local], [0, 3, 2], 1)
    has_p = (l.n_phone != "") & (r.n_phone != "")
    phone = np.select([~has_p, l.n_phone == r.n_phone], [0, 2], 1)
    jw = np.array([JaroWinkler.similarity(x, y) for x, y in zip(l.n_last, r.n_last)])
    last = np.select([(l.n_last == "") | (r.n_last == ""), l.n_last == r.n_last, jw >= 0.9], [0, 3, 2], 1)
    first = np.array([_first(x, y) for x, y in zip(l.n_first, r.n_first)])
    company = np.where(l.company == r.company, 2, 1)  # same resolved account (after account dedupe)
    has_t = l.title.notna() & r.title.notna()
    title = np.select([~has_t, l.title == r.title], [0, 2], 1)
    out = pd.DataFrame({"email": email, "phone": phone, "last_name": last, "first_name": first,
                        "company": company, "title": title})
    out.insert(0, "id_r", pairs.id_r.values)
    out.insert(0, "id_l", pairs.id_l.values)
    return out


def candidates(contacts: pd.DataFrame, account_cluster: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = prepare(contacts, account_cluster)
    return c, compare(c, candidate_pairs(c, "contact_id", BLOCKING_KEYS))
