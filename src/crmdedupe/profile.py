"""Data-quality profile of the raw CRM tables (what's broken before any dedupe)."""
import re

import pandas as pd

from . import normalize as N


def _phone_format(p) -> str:
    if not isinstance(p, str) or not p.strip():
        return "missing"
    return re.sub(r"\d", "9", p.strip())


def profile(accounts: pd.DataFrame, contacts: pd.DataFrame) -> dict:
    a, c = accounts, contacts
    pct = lambda n, d: round(100 * n / d, 1) if d else 0.0
    state_ok = a.billing_state.astype(str).str.fullmatch(r"[A-Z]{2}")
    email_n = c.email.map(N.email)
    return {
        "accounts": [
            ("Records", len(a), ""),
            ("Missing industry", a.industry.isna().sum(), f"{pct(a.industry.isna().sum(), len(a))}%"),
            ("Missing employee count", a.employee_count.isna().sum(), f"{pct(a.employee_count.isna().sum(), len(a))}%"),
            ("Missing website", a.website.isna().sum(), f"{pct(a.website.isna().sum(), len(a))}%"),
            ("State not a 2-letter code (e.g. 'Maryland', 'md')", (~state_ok).sum(), f"{pct((~state_ok).sum(), len(a))}%"),
            ("Website formats in use", a.website.dropna().map(lambda w: re.sub(r"[a-z0-9-]+\.[a-z]+$", "domain", w)).nunique(), ""),
        ],
        "contacts": [
            ("Records", len(c), ""),
            ("Invalid email", (email_n == "").sum(), f"{pct((email_n == '').sum(), len(c))}%"),
            ("Personal email (gmail etc.)", c.email.map(N.email_domain).isin(N.FREE_MAIL).sum(), ""),
            ("Email not lower-case", (c.email.fillna("") != c.email.fillna("").str.lower()).sum(), ""),
            ("Missing title", c.title.isna().sum(), f"{pct(c.title.isna().sum(), len(c))}%"),
            ("Distinct phone formats", c.phone.map(_phone_format).nunique(), ", ".join(c.phone.map(_phone_format).value_counts().index[:4])),
        ],
    }
