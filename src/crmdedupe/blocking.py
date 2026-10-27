"""Blocking: only compare records that share at least one cheap key, instead of all n² pairs."""
import itertools

import pandas as pd


def candidate_pairs(df: pd.DataFrame, id_col: str, keys: list[str], max_block: int = 200) -> pd.DataFrame:
    """Pairs (id_l < id_r) sharing any blocking key. Blocks larger than max_block are skipped
    (a key shared by hundreds of records, like a common surname, carries little information)."""
    seen = set()
    for key in keys:
        for _, ids in df[df[key].astype(bool)].groupby(key)[id_col]:
            ids = sorted(ids)
            if len(ids) < 2 or len(ids) > max_block:
                continue
            seen.update(itertools.combinations(ids, 2))
    return pd.DataFrame(sorted(seen), columns=["id_l", "id_r"])

