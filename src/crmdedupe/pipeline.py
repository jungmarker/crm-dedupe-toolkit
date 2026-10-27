"""End to end: match accounts → match contacts within resolved companies → survivorship → merge plans."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import accounts as A, contacts as Cn
from .cluster import clusters
from .labeling import review_sample, steward_labels
from .model import LinkageModel
from .survivorship import golden_accounts, pick_account_masters, pick_contact_masters

AUTO_MERGE = 0.90   # precision-first: merges above this go straight to the merge file
REVIEW = 0.50       # between REVIEW and AUTO_MERGE a data steward decides


def load(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    read = lambda p: pd.read_csv(p) if p.exists() else None
    return {"accounts": read(data_dir / "raw/accounts.csv"), "contacts": read(data_dir / "raw/contacts.csv"),
            "opportunities": read(data_dir / "raw/opportunities.csv"),
            "account_truth": read(data_dir / "truth/account_duplicates.csv"),
            "contact_truth": read(data_dir / "truth/contact_duplicates.csv")}


def _edges(levels, threshold):
    keep = levels[levels.match_probability >= threshold]
    return list(zip(keep.id_l, keep.id_r))


def train(d: dict, n_labels: int = 300, seed: int = 0) -> tuple[LinkageModel, LinkageModel, dict]:
    """Fit both models from a steward-reviewed sample of candidate pairs."""
    acols = [c.name for c in A.COMPARISONS]
    _, alv = A.candidates(d["accounts"])
    a_sample = review_sample(alv, A.COMPARISONS, n_labels, seed)
    a_sample["is_match"] = steward_labels(a_sample, d["accounts"].account_id, d["account_truth"],
                                          "duplicate_account_id", "master_account_id")
    am = LinkageModel(A.COMPARISONS).fit(a_sample[acols], a_sample.is_match)

    alv["match_probability"] = am.predict(alv[acols])
    acl = clusters(d["accounts"].account_id, _edges(alv, AUTO_MERGE))
    ccols = [c.name for c in Cn.COMPARISONS]
    _, clv = Cn.candidates(d["contacts"], acl)
    c_sample = review_sample(clv, Cn.COMPARISONS, n_labels, seed)
    c_sample["is_match"] = steward_labels(c_sample, d["contacts"].contact_id, d["contact_truth"],
                                          "duplicate_contact_id", "master_contact_id")
    cm = LinkageModel(Cn.COMPARISONS).fit(c_sample[ccols], c_sample.is_match)
    return am, cm, {"accounts": a_sample, "contacts": c_sample}


@dataclass
class Result:
    account_levels: pd.DataFrame
    contact_levels: pd.DataFrame
    account_merges: pd.DataFrame
    account_review: pd.DataFrame
    contact_merges: pd.DataFrame
    contact_review: pd.DataFrame
    golden: pd.DataFrame
    reparent: pd.DataFrame
    account_cluster: dict
    contact_cluster: dict


def _plan(ranked: pd.DataFrame, levels: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """One row per duplicate → master, with the strongest pair evidence linking it into the cluster."""
    dupes = ranked[ranked[id_col] != ranked.master_id]
    best = (pd.concat([levels.rename(columns={"id_l": "a", "id_r": "b"}),
                       levels.rename(columns={"id_l": "b", "id_r": "a"})])
            .sort_values("match_probability", ascending=False).drop_duplicates("a"))
    best = best[best.match_probability >= AUTO_MERGE].set_index("a")
    return pd.DataFrame({"master_id": dupes.master_id.values, "duplicate_id": dupes[id_col].values,
                         "match_probability": dupes[id_col].map(best.match_probability).round(3).values,
                         "evidence": dupes[id_col].map(best.evidence).values})


def _review(levels: pd.DataFrame, records: pd.DataFrame, id_col: str, show: list[str], cluster: dict):
    band = levels[(levels.match_probability >= REVIEW) & (levels.match_probability < AUTO_MERGE)]
    band = band[[cluster[l] != cluster[r] for l, r in zip(band.id_l, band.id_r)]]  # not already merged
    rec = records.set_index(id_col)[show]
    out = band[["id_l", "id_r", "match_probability", "evidence"]].copy()
    for col in show:
        out[f"{col}_l"] = rec.loc[band.id_l, col].values
        out[f"{col}_r"] = rec.loc[band.id_r, col].values
    return out.sort_values("match_probability", ascending=False).round({"match_probability": 3})


def run(d: dict, am: LinkageModel, cm: LinkageModel) -> Result:
    acc, con = d["accounts"], d["contacts"]
    acols, ccols = [c.name for c in A.COMPARISONS], [c.name for c in Cn.COMPARISONS]

    _, alv = A.candidates(acc)
    alv["match_probability"] = am.predict(alv[acols])
    alv["evidence"] = am.explain(alv[acols])
    acl = clusters(acc.account_id, _edges(alv, AUTO_MERGE))
    opp_counts = d["opportunities"].account_id.value_counts() if d["opportunities"] is not None else pd.Series()
    ranked_a = pick_account_masters(acc, acl, opp_counts)

    _, clv = Cn.candidates(con, acl)
    clv["match_probability"] = cm.predict(clv[ccols])
    clv["evidence"] = cm.explain(clv[ccols])
    ccl = clusters(con.contact_id, _edges(clv, AUTO_MERGE))
    ranked_c = pick_contact_masters(con, ccl)

    master_of = dict(zip(ranked_a.account_id, ranked_a.master_id))
    survivors = ranked_c[ranked_c.contact_id == ranked_c.master_id]
    moves = survivors[survivors.account_id.map(master_of) != survivors.account_id]
    reparent = pd.DataFrame({"contact_id": moves.contact_id, "from_account_id": moves.account_id,
                             "to_account_id": moves.account_id.map(master_of)})

    return Result(
        account_levels=alv, contact_levels=clv,
        account_merges=_plan(ranked_a, alv, "account_id"),
        account_review=_review(alv, acc, "account_id", ["account_name", "website", "billing_state", "employee_count"], acl),
        contact_merges=_plan(ranked_c, clv, "contact_id"),
        contact_review=_review(clv, con, "contact_id", ["first_name", "last_name", "email", "phone", "title"], ccl),
        golden=golden_accounts(ranked_a), reparent=reparent, account_cluster=acl, contact_cluster=ccl)
