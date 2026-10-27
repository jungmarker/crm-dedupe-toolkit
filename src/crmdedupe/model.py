"""Pair classifier: logistic regression on comparison levels, stored as plain JSON.

Every comparison level becomes a 0/1 feature, so each coefficient reads directly as
"how much does this kind of agreement or disagreement move the log-odds of a match".
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Comparison:
    name: str
    levels: list[str]           # label for each level value (index = level)
    null_level: int | None = 0  # baseline level ("missing" or "different"): no feature, contributes 0


def one_hot(levels: pd.DataFrame, comparisons: list[Comparison]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for c in comparisons:
        x = levels[c.name].to_numpy()
        for v, label in enumerate(c.levels):
            if v != c.null_level:
                cols.append((x == v).astype(float))
                names.append(f"{c.name}: {label}")
    return np.column_stack(cols), names


class LinkageModel:
    def __init__(self, comparisons: list[Comparison], coef=None, intercept: float = 0.0):
        self.comparisons = comparisons
        self.coef = None if coef is None else np.asarray(coef, float)
        self.intercept = intercept

    def fit(self, levels: pd.DataFrame, y, C: float = 10.0) -> "LinkageModel":
        from sklearn.linear_model import LogisticRegression
        X, _ = one_hot(levels, self.comparisons)
        lr = LogisticRegression(C=C, max_iter=2000).fit(X, np.asarray(y, int))
        self.coef, self.intercept = lr.coef_[0], float(lr.intercept_[0])
        return self

    def predict(self, levels: pd.DataFrame) -> np.ndarray:
        X, _ = one_hot(levels, self.comparisons)
        return 1 / (1 + np.exp(-(X @ self.coef + self.intercept)))

    def coef_table(self) -> pd.DataFrame:
        _, names = one_hot(pd.DataFrame({c.name: [0] for c in self.comparisons}), self.comparisons)
        return pd.DataFrame({"feature": names, "log_odds": self.coef.round(2)})

    def explain(self, levels: pd.DataFrame, top: int = 3) -> list[str]:
        """Strongest evidence for each pair, e.g. 'domain: same (+3.1); state: different (-2.0)'."""
        X, names = one_hot(levels, self.comparisons)
        contrib = X * self.coef
        out = []
        for row in contrib:
            idx = np.argsort(-np.abs(row))[:top]
            out.append("; ".join(f"{names[i]} ({row[i]:+.1f})" for i in idx if row[i] != 0))
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"intercept": self.intercept, "coef": dict(zip(self.coef_table().feature,
                                                                                  self.coef.tolist()))}, indent=1))

    @classmethod
    def load(cls, path: Path, comparisons: list[Comparison]) -> "LinkageModel":
        d = json.loads(Path(path).read_text())
        m = cls(comparisons, intercept=d["intercept"])
        m.coef = np.array([d["coef"][f] for f in m_names(comparisons)])
        return m


def m_names(comparisons):
    return [f"{c.name}: {label}" for c in comparisons for v, label in enumerate(c.levels) if v != c.null_level]
