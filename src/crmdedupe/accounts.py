"""Account (company) matching."""
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import token_set_ratio

from . import normalize as N
from .blocking import candidate_pairs
from .model import Comparison

COMPARISONS = [
    Comparison("domain", ["missing", "different", "same"]),
    Comparison("name", ["different", "similar", "very similar", "same root name", "identical name"]),
    Comparison("state", ["missing", "different", "same"]),
    Comparison("industry", ["missing", "different", "same"]),
    Comparison("employees", ["missing", "different", "within 30%", "exact"]),
    Comparison("owner", ["missing", "different", "same"]),
]
BLOCKING_KEYS = ["n_domain", "n_name", "k_prefix_state"]


def prepare(accounts: pd.DataFrame) -> pd.DataFrame:
    a = accounts.copy()
    a["n_name"] = a["account_name"].map(N.company_name)
    a["n_full"] = a["account_name"].fillna("").str.lower()
    a["n_domain"] = a["website"].map(N.domain)
    a["n_state"] = a["billing_state"].map(N.state)
    a["k_prefix_state"] = a["n_name"].str[:4] + "|" + a["n_state"]
    return a


def compare(a: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    l = a.set_index("account_id").loc[pairs.id_l].reset_index(drop=True)
    r = a.set_index("account_id").loc[pairs.id_r].reset_index(drop=True)

    def eq(col):
        both = l[col].notna() & r[col].notna() & (l[col].astype(str) != "") & (r[col].astype(str) != "")
        return np.where(~both, 0, np.where(l[col].astype(str) == r[col].astype(str), 2, 1))

    jw = np.array([JaroWinkler.similarity(x, y) for x, y in zip(l.n_name, r.n_name)])
    tsr = np.array([token_set_ratio(x, y) / 100 for x, y in zip(l.n_full, r.n_full)])
    full_l = l.n_full.str.replace(r"^the\s+|[^a-z0-9]", "", regex=True)
    full_r = r.n_full.str.replace(r"^the\s+|[^a-z0-9]", "", regex=True)
    name = np.select([full_l == full_r, l.n_name == r.n_name, jw >= 0.95, (jw >= 0.85) | (tsr >= 0.9)], [4, 3, 2, 1], 0)
    ec_l, ec_r = l.employee_count.astype(float), r.employee_count.astype(float)
    rel = (ec_l - ec_r).abs() / np.maximum(ec_l, ec_r)
    emp = np.select([ec_l.isna() | ec_r.isna(), ec_l == ec_r, rel <= 0.30], [0, 3, 2], 1)
    out = pd.DataFrame({"domain": eq("n_domain"), "name": name, "state": eq("n_state"), "industry": eq("industry"),
                        "employees": emp, "owner": eq("owner_id")})
    out.insert(0, "id_r", pairs.id_r.values)
    out.insert(0, "id_l", pairs.id_l.values)
    out["name_jw"] = jw.round(3)
    return out


def candidates(accounts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepared accounts and the comparison levels of every blocked candidate pair."""
    a = prepare(accounts)
    return a, compare(a, candidate_pairs(a, "account_id", BLOCKING_KEYS))
