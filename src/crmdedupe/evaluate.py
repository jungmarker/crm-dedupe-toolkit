"""Score predicted clusters against ground truth (pairwise precision / recall / F1)."""
import itertools

import pandas as pd

from .cluster import clusters


def _pairs_from_clusters(assign: dict) -> set:
    groups = {}
    for rid, cid in assign.items():
        groups.setdefault(cid, []).append(rid)
    return {frozenset(p) for g in groups.values() if len(g) > 1 for p in itertools.combinations(sorted(g), 2)}


def truth_pairs(ids, truth: pd.DataFrame, dup_col: str, master_col: str) -> set:
    return _pairs_from_clusters(clusters(ids, zip(truth[dup_col], truth[master_col])))


def score(ids, predicted_pairs, true_pairs: set) -> dict:
    pred = _pairs_from_clusters(clusters(ids, predicted_pairs))
    tp = len(pred & true_pairs)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(true_pairs) if true_pairs else 1.0
    return dict(predicted_pairs=len(pred), true_pairs=len(true_pairs), true_positives=tp,
                false_positives=len(pred) - tp, missed=len(true_pairs) - tp,
                precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r) if p + r else 0, 4))


def evaluate(d: dict, result, auto: float, review: float) -> dict:
    """Score the pipeline and simple baselines against ground truth."""
    from . import normalize as N

    out = {}
    for kind, id_col, records, truth, levels, dup, mas in (
        ("accounts", "account_id", d["accounts"], d["account_truth"], result.account_levels,
         "duplicate_account_id", "master_account_id"),
        ("contacts", "contact_id", d["contacts"], d["contact_truth"], result.contact_levels,
         "duplicate_contact_id", "master_contact_id")):
        ids = records[id_col]
        tp = truth_pairs(ids, truth, dup, mas)
        pairs = lambda lo: list(zip(*[levels[levels.match_probability >= lo][c] for c in ("id_l", "id_r")]))
        band = levels[(levels.match_probability >= review) & (levels.match_probability < auto)]
        accepted = [p for p in zip(band.id_l, band.id_r) if frozenset(p) in tp]  # steward accepts true matches
        rows = {"Auto-merge only (p ≥ %.2f)" % auto: score(ids, pairs(auto), tp),
                "After steward review (p ≥ %.2f)" % review: score(ids, pairs(auto) + accepted, tp)}
        if kind == "accounts":
            key_name = records.account_name.map(N.company_name)
            key_domain = records.website.map(N.domain)
            rows["Baseline: same cleaned name"] = score(ids, _same_key(records[id_col], key_name), tp)
            rows["Baseline: same website domain"] = score(ids, _same_key(records[id_col], key_domain), tp)
        else:
            rows["Baseline: same email (case-insensitive)"] = score(ids, _same_key(records[id_col], records.email.map(N.email)), tp)
            rows["Baseline: same first + last name"] = score(
                ids, _same_key(records[id_col], records.first_name.str.lower() + " " + records.last_name.str.lower()), tp)
        merges = result.account_merges if kind == "accounts" else result.contact_merges
        truth_master = dict(zip(truth[dup], truth[mas]))
        found = merges[merges.duplicate_id.isin(truth_master)]
        correct = (found.duplicate_id.map(truth_master) == found.master_id).sum()
        out[kind] = {"records": len(records), "true_duplicates": len(truth), "scores": rows,
                     "survivor_correct": f"{correct}/{len(found)}" if len(found) else "n/a",
                     "review_queue": int(len(result.account_review if kind == "accounts" else result.contact_review))}
    return out


def _same_key(ids: pd.Series, key: pd.Series) -> list:
    df = pd.DataFrame({"id": ids.values, "k": key.values})
    df = df[df.k.astype(bool) & df.k.notna()]
    return [(a, b) for _, g in df.groupby("k").id for a, b in itertools.combinations(sorted(g), 2)]
