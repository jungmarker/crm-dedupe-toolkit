"""Single-file HTML report: data-quality profile, dedupe results, accuracy and review queue."""
import html

import pandas as pd

CSS = """:root{--bg:#f7f8fa;--card:#fff;--ink:#14171f;--muted:#5f6675;--line:#e4e7ec;--a:#2f5bea;--good:#18a07a;--bad:#d64545}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b24;--ink:#e8ebf1;--muted:#99a1b3;--line:#262c38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 16px 48px}h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--muted);margin:4px 0 22px}.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.two{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
.kv{font-size:24px;font-weight:650;margin-top:4px}.kl{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.ks{color:var(--muted);font-size:12px}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}td:last-child{min-width:260px}th{color:var(--muted);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}.pos{color:var(--good)}.neg{color:var(--bad)}
p.note{color:var(--muted);font-size:12.5px;margin:8px 0 0}"""


def _table(df: pd.DataFrame, num=()) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(
        f'<td class="{"n" if c in num else ""}">{html.escape("" if pd.isna(v) else str(v))}</td>'
        for c, v in row.items()) + "</tr>" for _, row in df.iterrows())
    return f"<table><tr>{head}</tr>{body}</table>"


def _coef(df: pd.DataFrame) -> str:
    rows = "".join(f'<tr><td>{html.escape(f)}</td><td class="n {"pos" if w > 0 else "neg"}">{w:+.2f}</td></tr>'
                   for f, w in zip(df.feature, df.log_odds))
    return f"<table><tr><th>Evidence</th><th>Log-odds</th></tr>{rows}</table>"


def build(result, prof: dict, metrics: dict | None, am, cm, title: str) -> str:
    acc_dupes, con_dupes = len(result.account_merges), len(result.contact_merges)
    tiles = [("Duplicate accounts merged", acc_dupes, f"{len(result.account_review)} pairs sent to review"),
             ("Duplicate contacts merged", con_dupes, f"{len(result.contact_review)} pairs sent to review"),
             ("Contacts re-parented", len(result.reparent), "moved off duplicate accounts"),
             ("Golden account records", len(result.golden), "one per real company")]
    if metrics:
        s = metrics["accounts"]["scores"]
        auto = next(v for k, v in s.items() if k.startswith("Auto"))
        tiles.append(("Auto-merge precision", f"{auto['precision']:.0%}", f"accounts · recall {auto['recall']:.0%}"))
    t = "".join(f'<div class="card"><div class="kl">{a}</div><div class="kv">{b}</div><div class="ks">{c}</div></div>'
                for a, b, c in tiles)
    prof_html = "".join(f'<div class="card"><h2>Data quality · {k}</h2>'
                        + _table(pd.DataFrame(v, columns=["Check", "Count", "Detail"]), num={"Count"}) + "</div>"
                        for k, v in prof.items())
    acc_html = ""
    if metrics:
        for kind in ("accounts", "contacts"):
            m = metrics[kind]
            df = pd.DataFrame([{"Method": k, "Precision": f"{v['precision']:.1%}", "Recall": f"{v['recall']:.1%}",
                                "F1": f"{v['f1']:.3f}", "False merges": v["false_positives"], "Missed": v["missed"]}
                               for k, v in m["scores"].items()])
            acc_html += (f'<div class="card"><h2>Accuracy vs. ground truth · {kind}</h2>'
                         + _table(df, num={"Precision", "Recall", "F1", "False merges", "Missed"})
                         + f'<p class="note">Pairwise, on {m["true_duplicates"]} known duplicates. '
                           f'Survivor chosen correctly: {m["survivor_correct"]}.</p></div>')
    review = result.account_review.head(12)[["match_probability", "account_name_l", "account_name_r", "website_l",
                                             "website_r", "billing_state_l", "billing_state_r", "evidence"]]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CRM Dedupe Report</title><style>{CSS}</style></head>
<body><main><h1>{html.escape(title)}</h1><div class="sub">Duplicate detection, survivorship and merge plan</div>
<div class="grid">{t}</div><div class="two">{prof_html}</div><div class="two">{acc_html}</div>
<div class="two"><div class="card"><h2>What the account model learned</h2>{_coef(am.coef_table())}
<p class="note">Logistic regression on comparison levels, trained on a steward-reviewed sample. Positive = evidence of the same company.</p></div>
<div class="card"><h2>What the contact model learned</h2>{_coef(cm.coef_table())}</div></div>
<div class="card" style="margin-top:14px"><h2>Account review queue (top 12)</h2>{_table(review.rename(columns=lambda c: c.replace("_l", " (A)").replace("_r", " (B)").replace("_", " ")))}
<p class="note">Pairs between 50% and 90% match probability. A steward approves or rejects each one; nothing here merges automatically.</p></div>
</main></body></html>"""
