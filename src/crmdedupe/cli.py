"""crm-dedupe train | run"""
import argparse
import json
from pathlib import Path

from . import accounts as A, contacts as Cn
from .evaluate import evaluate
from .model import LinkageModel
from .pipeline import AUTO_MERGE, REVIEW, load, run, train
from .profile import profile
from .report import build


def main(argv=None):
    p = argparse.ArgumentParser(prog="crm-dedupe", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit models from a steward-reviewed sample of candidate pairs")
    t.add_argument("--data", required=True, type=Path)
    t.add_argument("--models", default=Path("models"), type=Path)
    t.add_argument("--labels", default=300, type=int, help="pairs per model to review")
    r = sub.add_parser("run", help="find duplicates and write merge plans + report")
    r.add_argument("--data", required=True, type=Path)
    r.add_argument("--models", default=Path("models"), type=Path)
    r.add_argument("--out", default=Path("out"), type=Path)
    args = p.parse_args(argv)

    d = load(args.data)
    if args.cmd == "train":
        am, cm, samples = train(d, args.labels)
        args.models.mkdir(parents=True, exist_ok=True)
        am.save(args.models / "accounts.json")
        cm.save(args.models / "contacts.json")
        for k, s in samples.items():
            s.to_csv(args.models / f"labeled_{k}.csv", index=False)
            print(f"{k}: {len(s)} reviewed pairs ({int(s.is_match.sum())} matches) → {args.models}/{k}.json")
        return

    am = LinkageModel.load(args.models / "accounts.json", A.COMPARISONS)
    cm = LinkageModel.load(args.models / "contacts.json", Cn.COMPARISONS)
    res = run(d, am, cm)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res.account_merges.to_csv(out / "account_merge_plan.csv", index=False)
    res.account_review.to_csv(out / "account_review_queue.csv", index=False)
    res.contact_merges.to_csv(out / "contact_merge_plan.csv", index=False)
    res.contact_review.to_csv(out / "contact_review_queue.csv", index=False)
    res.reparent.to_csv(out / "contact_reparent.csv", index=False)
    res.golden.to_csv(out / "golden_accounts.csv", index=False)
    metrics = None
    if d["account_truth"] is not None:
        metrics = evaluate(d, res, AUTO_MERGE, REVIEW)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "report.html").write_text(build(res, profile(d["accounts"], d["contacts"]), metrics, am, cm,
                                           f"CRM Dedupe Report · {args.data.name}"))
    print(f"accounts: {len(res.account_merges)} merges, {len(res.account_review)} to review | "
          f"contacts: {len(res.contact_merges)} merges, {len(res.contact_review)} to review | "
          f"{len(res.reparent)} contacts re-parented → {out}/")
    if metrics:
        for kind in ("accounts", "contacts"):
            for name, s in metrics[kind]["scores"].items():
                print(f"  {kind:9s} {name:42s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
            print(f"  {kind:9s} survivor correct: {metrics[kind]['survivor_correct']}")


if __name__ == "__main__":
    main()
