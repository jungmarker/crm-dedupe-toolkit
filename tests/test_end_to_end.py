"""Train on one synthetic company, evaluate on a different one (held out)."""
import pytest

from crmdedupe.evaluate import evaluate
from crmdedupe.pipeline import AUTO_MERGE, REVIEW, run, train

ledgerly = pytest.importorskip("ledgerly.generate")


def as_input(seed):
    d = ledgerly.build(seed=seed)
    raw, truth = d["raw"], d["truth"]
    return {"accounts": raw["accounts"], "contacts": raw["contacts"], "opportunities": raw["opportunities"],
            "account_truth": truth["account_duplicates"], "contact_truth": truth["contact_duplicates"]}


@pytest.fixture(scope="module")
def holdout():
    am, cm, _ = train(as_input(42), 400)
    d = as_input(7)
    res = run(d, am, cm)
    return d, res, evaluate(d, res, AUTO_MERGE, REVIEW)


def test_auto_merges_are_precise(holdout):
    _, _, m = holdout
    for kind in ("accounts", "contacts"):
        auto = next(v for k, v in m[kind]["scores"].items() if k.startswith("Auto"))
        assert auto["precision"] >= 0.98, kind


def test_review_recovers_most_duplicates(holdout):
    _, _, m = holdout
    for kind in ("accounts", "contacts"):
        after = next(v for k, v in m[kind]["scores"].items() if k.startswith("After"))
        assert after["recall"] >= 0.9, kind


def test_beats_domain_baseline(holdout):
    s = holdout[2]["accounts"]["scores"]
    after = next(v for k, v in s.items() if k.startswith("After"))
    assert after["f1"] > s["Baseline: same website domain"]["f1"] + 0.2


def test_merge_plan_integrity(holdout):
    d, res, _ = holdout
    mp = res.account_merges
    assert mp.duplicate_id.is_unique
    assert not set(mp.duplicate_id) & set(mp.master_id)          # a master is never merged away
    assert len(res.golden) == len(d["accounts"]) - len(mp)        # one golden row per surviving company
    assert set(res.reparent.to_account_id) <= set(mp.master_id)   # contacts only move onto masters
    survivors = d["accounts"].set_index("account_id").loc[mp.master_id]
    dupes = d["accounts"].set_index("account_id").loc[mp.duplicate_id]
    rank = {"Customer": 0, "Churned": 1, "Prospect": 2}
    assert (survivors.status.map(rank).values <= dupes.status.map(rank).values).all()  # customers survive
