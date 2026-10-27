"""Build a small labeled training set the way a data steward would: review a sample of candidate pairs.

The sample is stratified by a crude agreement count so the steward sees obvious matches, obvious
non-matches and, mostly, the ambiguous middle, where labels are worth the most.
"""
import numpy as np
import pandas as pd

from .evaluate import truth_pairs


def review_sample(levels: pd.DataFrame, comparisons, n: int = 300, seed: int = 0) -> pd.DataFrame:
    agree = sum((levels[c.name] == len(c.levels) - 1).astype(int) for c in comparisons)
    rng = np.random.default_rng(seed)
    strata = pd.qcut(agree.rank(method="first"), 5, labels=False)
    share = {0: 0.1, 1: 0.15, 2: 0.25, 3: 0.25, 4: 0.25}  # oversample the middle and top
    picks = []
    for s, frac in share.items():
        idx = levels.index[strata == s]
        k = min(len(idx), int(round(n * frac)))
        picks.extend(rng.choice(idx, size=k, replace=False))
    return levels.loc[sorted(picks)].copy()


def steward_labels(sample: pd.DataFrame, ids, truth: pd.DataFrame, dup_col: str, master_col: str) -> pd.Series:
    """Stand-in for human review: label each sampled pair from the hidden ground truth."""
    tp = truth_pairs(ids, truth, dup_col, master_col)
    return pd.Series([frozenset((a, b)) in tp for a, b in zip(sample.id_l, sample.id_r)], index=sample.index,
                     name="is_match").astype(int)
