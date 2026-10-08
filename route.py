#!/usr/bin/env python3
"""Kestrel Home: service-request routing classifier.

Usage:  python route.py [--data data] [--out out]

What it does
  1. Cleans the export (renamed teams merged, Zoho mojibake repaired, Zoho UTC resolve times fixed).
  2. Trains on where each request ACTUALLY ENDED UP (final_team), not on the bot's queue,
     because the bot's queue is wrong for about 1 request in 4.
  3. Scores honestly on a time-ordered hold-out (Apr-Jun 2026), after picking the
     confidence threshold on an earlier window (Jan-Mar 2026).
  4. Writes predictions for test_unlabelled.csv; low-confidence rows are flagged for triage.
  5. Prices the change with the policy's own cost figures and shows workload by team.

No API keys, no network, no per-request cost. A full run takes a few seconds.
"""
import argparse, json, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support

RENAME = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}  # ops-policy s5
COST_TRANSFER, COST_EXTRA_CONTACT, LICENCE_PER_YEAR = 305, 260, 320_000                # ops-policy s4
VAL_START, TEST_START = "2026-01-01", "2026-04-01"
TARGET_ACC_ROUTED = 0.95        # business rule: auto-route only where we expect >= 95% right
SEED = 7


def fix_text(s: str) -> str:
    """Zoho mojibake ('thÃ©', 'Ã¢â‚¬Â¦') -> plain lowercase ASCII; strips reference numbers."""
    s = str(s)
    for _ in range(2):                      # some rows are double-encoded
        try:
            s = s.encode("cp1252").decode("utf-8")
        except Exception:
            break
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return pd.Series([s]).str.replace(r"\b(?:sr|ko)\d+\b", " ", regex=True).iloc[0]


def load(data: Path):
    need = ["train.csv", "test_unlabelled.csv", "resolution_log.csv"]
    miss = [f for f in need if not (data / f).exists()]
    if miss:
        sys.exit(f"Missing {miss} in {data}/. Put the Kestrel CSVs there (see README).")
    tr = pd.read_csv(data / "train.csv")
    te = pd.read_csv(data / "test_unlabelled.csv")
    rl = pd.read_csv(data / "resolution_log.csv")
    m = tr.merge(rl, on="request_id", validate="1:1")
    m["bot"] = m.team_label.replace(RENAME)
    m["first"] = m.first_team.replace(RENAME)
    m["final"] = m.final_team.replace(RENAME)
    m["created"] = pd.to_datetime(m.created_at_ist)
    m["resolved"] = pd.to_datetime(m.resolved_at)
    # Zoho resolve events are UTC and were never converted (ops-policy s9) -> +5:30 to get IST
    z = m.source == "legacy_zoho"
    m.loc[z, "resolved"] = m.loc[z, "resolved"] + pd.Timedelta(hours=5, minutes=30)
    m["hours_to_resolve"] = (m.resolved - m.created).dt.total_seconds() / 3600
    for d in (m, te):
        d["txt"] = d.request_text.map(fix_text)
    te["created"] = pd.to_datetime(te.created_at_ist)
    return m, te


def make_model():
    feats = ColumnTransformer([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "txt"),
        ("c", OneHotEncoder(handle_unknown="ignore"), ["channel", "product_family", "warranty_status"]),
    ])
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=5.0, random_state=SEED))


def predict(model, df):
    P = model.predict_proba(df)
    order = np.argsort(-P, axis=1)
    cls = model.classes_
    return cls[order[:, 0]], P[np.arange(len(df)), order[:, 0]], cls[order[:, 1]]


def pick_threshold(conf, correct):
    """Lowest threshold at which the auto-routed share is >= TARGET_ACC_ROUTED accurate."""
    for t in np.arange(0.40, 0.96, 0.02):
        sel = conf >= t
        if sel.sum() >= 50 and correct[sel].mean() >= TARGET_ACC_ROUTED:
            return round(float(t), 2)
    return 0.90


def cost_of(transfers):
    transfers = np.asarray(transfers)
    return float((transfers * COST_TRANSFER).sum() + (transfers > 0).sum() * COST_EXTRA_CONTACT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m, te = load(data)
    R = {}

    # ---------- 0. what the client asked for vs what is true ----------
    R["bot_agrees_with_final_all"] = float((m.bot == m.final).mean())
    R["requests_with_transfers_pct"] = float((m.transfers > 0).mean())
    R["transfers_total"] = int(m.transfers.sum())

    # ---------- 1. time-ordered evaluation ----------
    A = m[m.created < VAL_START]; B = m[(m.created >= VAL_START) & (m.created < TEST_START)]; C = m[m.created >= TEST_START]
    # threshold chosen on B using a model that has only seen A
    mdl = make_model().fit(A, A.final)
    pB, cB, _ = predict(mdl, B)
    thr = pick_threshold(cB, pB == B.final.values)
    # final hold-out C: model sees A+B only
    mdl = make_model().fit(pd.concat([A, B]), pd.concat([A, B]).final)
    pC, cC, p2C = predict(mdl, C)
    C = C.assign(pred=pC, conf=cC, second=p2C)
    C["routed"] = C.conf >= thr
    ok = C.pred == C.final
    R.update(threshold=thr, holdout_n=len(C), holdout_months="Apr-Jun 2026",
             bot_acc_holdout=float((C.bot == C.final).mean()),
             model_acc_all=float(ok.mean()),
             auto_route_share=float(C.routed.mean()),
             model_acc_when_routed=float(ok[C.routed].mean()),
             bot_acc_on_same_routed=float((C.bot == C.final)[C.routed].mean()),
             bot_acc_on_triage=float((C.bot == C.final)[~C.routed].mean()),
             model_acc_on_triage=float(ok[~C.routed].mean()))
    # the 90% bar the client set, measured against the thing they named
    mb = make_model().fit(pd.concat([A, B]), pd.concat([A, B]).bot)
    pb, _, _ = predict(mb, C)
    R["bot_clone_matches_bot"] = float((pb == C.bot.values).mean())
    R["bot_clone_matches_final"] = float((pb == C.final.values).mean())

    pr, rc, f1, sup = precision_recall_fscore_support(C.final, C.pred, labels=sorted(C.final.unique()), zero_division=0)
    per = pd.DataFrame({"team": sorted(C.final.unique()), "precision": pr.round(3), "recall": rc.round(3), "f1": f1.round(3), "n": sup})
    per.to_csv(out / "per_team_holdout.csv", index=False)
    pd.crosstab(C.final, C.pred, rownames=["actual_final_team"], colnames=["predicted"]).to_csv(out / "confusion_holdout.csv")
    C[["request_id", "created_at_ist", "final", "bot", "pred", "conf", "routed", "request_text"]].to_csv(out / "holdout_errors_and_all.csv", index=False)

    # ---------- 2. cost model on the hold-out (conservative) ----------
    avg_t = m.loc[m.transfers > 0, "transfers"].mean()
    cost_bot = cost_of(C.transfers)
    wrong_cost = avg_t * COST_TRANSFER + COST_EXTRA_CONTACT       # what a typical misroute costs today
    # model: routed & right = 0; routed & wrong = typical misroute; sent to triage = still pays a typical misroute (conservative)
    n_routed_wrong = int((C.routed & ~ok).sum()); n_triage = int((~C.routed).sum())
    cost_model = (n_routed_wrong + n_triage) * wrong_cost
    months = 3
    R.update(avg_transfers_per_misroute=float(avg_t), cost_per_misroute_inr=float(wrong_cost),
             holdout_cost_bot_inr=cost_bot, holdout_cost_model_inr=float(cost_model),
             annual_transfer_cost_saving_inr=float((cost_bot - cost_model) * 12 / months),
             annual_licence_inr=LICENCE_PER_YEAR,
             annual_total_inr=float((cost_bot - cost_model) * 12 / months + LICENCE_PER_YEAR),
             transferred_share_bot=float((C.transfers > 0).mean()),
             transferred_share_model_conservative=float((n_routed_wrong + n_triage) / len(C)),
             transferred_share_model_if_triage_fixed=float(n_routed_wrong / len(C)),
             # optimistic case: one clarifying contact (Rs 260) resolves each triage row, no transfers after it
             annual_saving_if_triage_fixed_inr=float((cost_bot - n_routed_wrong * wrong_cost - n_triage * COST_EXTRA_CONTACT) * 12 / months + LICENCE_PER_YEAR))

    # ---------- 3. final model on everything -> predictions for the unlabelled set ----------
    full = make_model().fit(m, m.final)
    pt, ct, p2 = predict(full, te)
    det = pd.DataFrame({"request_id": te.request_id, "team": pt, "confidence": ct.round(3), "second_choice": p2,
                        "needs_triage": ct < thr, "request_text": te.request_text})
    det.to_csv(out / "predictions_detail.csv", index=False)
    det[["request_id", "team"]].to_csv(out / "submission.csv", index=False)
    R["test_rows"] = len(det); R["test_triage_share"] = float(det.needs_triage.mean())

    # ---------- 4. workload (headcount question) ----------
    m["month"] = m.created.dt.to_period("M").astype(str)
    w = m.groupby("final").agg(requests=("request_id", "count"), transfers_on_requests=("transfers", "sum"),
                               median_hours_to_resolve=("hours_to_resolve", "median")).round(1)
    w["share_pct"] = (w.requests / w.requests.sum() * 100).round(1)
    w["requests_per_month"] = (w.requests / 15).round(0)
    sent = m.groupby("first").agg(bot_queue_requests=("request_id", "count"), sent_out_misrouted=("transfers", lambda s: (s > 0).sum()))
    w = w.join(sent).sort_values("requests", ascending=False)
    w["bot_queue_vs_actual_pct"] = ((w.bot_queue_requests / w.requests - 1) * 100).round(0)
    w.to_csv(out / "workload_by_team.csv")
    tm = det.groupby("team").size().rename("predicted_test_requests").to_frame()
    tm["per_month"] = (tm.predicted_test_requests / 3).round(0)
    tm.to_csv(out / "workload_forecast_jul_sep_2026.csv")

    # ---------- 5. where the bot goes wrong (the actual root cause) ----------
    mis = m[m.first != m.final]
    pat = pd.crosstab(mis["first"], mis["final"])
    pat.to_csv(out / "bot_misroute_matrix.csv")
    paid = m.txt.str.contains(r"\b(?:paid|payment|emi)\b")
    bill = m[m.first == "Billing"]
    R["billing_queue_correct_pct"] = float((bill.final == "Billing").mean())
    R["billing_queue_with_paid_wording_but_not_billing_pct"] = float(((bill.final != "Billing") & paid[bill.index]).sum() / len(bill))
    cons = m[m.first == "Filters & Consumables"]
    R["consumables_queue_actually_repairs_pct"] = float((cons.final == "Repairs").mean())

    json.dump(R, open(out / "metrics.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print("== Bot vs reality:     bot queue = final team on %.1f%% of requests; %.1f%% needed >=1 transfer" % (100 * R["bot_agrees_with_final_all"], 100 * R["requests_with_transfers_pct"]))
    print("== Client's 90%% test:   a clone of the bot matches the bot %.1f%%, but matches the real final team only %.1f%%" % (100 * R["bot_clone_matches_bot"], 100 * R["bot_clone_matches_final"]))
    print("== Hold-out (Apr-Jun 2026, n=%d, trained on earlier data only)" % len(C))
    print("   bot correct            : %.1f%%" % (100 * R["bot_acc_holdout"]))
    print("   model correct (all)    : %.1f%%" % (100 * R["model_acc_all"]))
    print("   auto-route at conf>=%.2f: %.1f%% of requests, %.1f%% correct (bot on the same rows: %.1f%%)" % (thr, 100 * R["auto_route_share"], 100 * R["model_acc_when_routed"], 100 * R["bot_acc_on_same_routed"]))
    print("   triage (vague text)    : %.1f%% of requests; model %.1f%% / bot %.1f%% correct there" % (100 * (1 - R["auto_route_share"]), 100 * R["model_acc_on_triage"], 100 * R["bot_acc_on_triage"]))
    print("== Money (policy costs; conservative: triage rows still pay a misroute)")
    print("   transferred share  bot %.1f%% -> model %.1f%% (%.1f%% if one clarifying question fixes every triage row)" % (100 * R["transferred_share_bot"], 100 * R["transferred_share_model_conservative"], 100 * R["transferred_share_model_if_triage_fixed"]))
    print("   optimistic case    Rs %s a year in total (licence included)" % f"{R['annual_saving_if_triage_fixed_inr']:,.0f}")
    print("   saving per year    Rs %s transfers + Rs %s licence = Rs %s" % (f"{R['annual_transfer_cost_saving_inr']:,.0f}", f"{LICENCE_PER_YEAR:,}", f"{R['annual_total_inr']:,.0f}"))
    print("== Test predictions: %d rows, %.1f%% flagged for triage -> %s" % (len(det), 100 * R["test_triage_share"], out / "submission.csv"))
    print("\nPer-team hold-out:"); print(per.to_string(index=False))


if __name__ == "__main__":
    main()
