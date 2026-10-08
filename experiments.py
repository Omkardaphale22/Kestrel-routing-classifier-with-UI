#!/usr/bin/env python3
"""Model-choice log: every variant tried, scored on the same hold-out (Apr-Jun 2026), and what was dropped.
Run after route.py.  Writes out/experiments.csv.  Deterministic, no network."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from route import load, VAL_START, TEST_START
from pathlib import Path

m, te = load(Path("data"))
fit = m[m.created < TEST_START]; ho = m[m.created >= TEST_START].copy()
CAT = ["channel", "product_family", "warranty_status"]

def feats(word=True, cat=True, char=False, col="txt"):
    parts = []
    if word: parts.append(("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), col))
    if char: parts.append(("ch", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), col))
    if cat: parts.append(("c", OneHotEncoder(handle_unknown="ignore"), CAT))
    return ColumnTransformer(parts)

rows = []
def add(name, decision, pred=None, conf=None, why=""):
    r = {"variant": name, "kept_or_dropped": decision, "acc_vs_final": np.nan, "acc_vs_bot": np.nan, "acc_at_85pct_coverage": np.nan, "why": why}
    if pred is not None:
        r["acc_vs_final"] = round(float((pred == ho.final.values).mean()), 3)
        r["acc_vs_bot"] = round(float((pred == ho.bot.values).mean()), 3)
        if conf is not None:
            k = int(0.85 * len(ho)); idx = np.argsort(-conf)[:k]
            r["acc_at_85pct_coverage"] = round(float((pred[idx] == ho.final.values[idx]).mean()), 3)
    rows.append(r)

def fitpred(clf, feat, target="final", df=fit):
    p = make_pipeline(feat, clf).fit(df, df[target])
    if hasattr(p, "predict_proba"):
        P = p.predict_proba(ho); return p.classes_[P.argmax(1)], P.max(1)
    d = p.decision_function(ho); return p.classes_[d.argmax(1)], d.max(1)

maj = fit.final.value_counts().idxmax()
add("0. always predict the biggest team (Repairs)", "dropped", np.array([maj] * len(ho)), None, "floor for scale")
add("1. the vendor bot as it is today", "baseline", ho.bot.values, None, "what we are replacing")
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats(), target="bot")
add("2. LR trained on the bot's queue (client's brief)", "dropped", p, c, "hits 90%+ vs the bot but copies its mistakes; real accuracy ~76%")
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats(col="request_text"))
add("3. LR final team, raw text (no mojibake repair)", "dropped", p, c, "repair is cheap; see delta to row 6")
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats(cat=False))
add("4. LR final team, text only", "dropped", p, c, "channel/product/warranty add a little")
p, c = fitpred(ComplementNB(alpha=0.3), feats())
add("5. Complement Naive Bayes", "dropped", p, c, "worse and badly calibrated")
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats())
add("6. LR final team, word 1-2grams + channel/product/warranty (CHOSEN)", "KEPT", p, c, "best calibrated; fast; explainable")
p, c = fitpred(LinearSVC(C=0.5), feats())
add("7. Linear SVM", "dropped", p, c, "same accuracy within noise; gives no probabilities, so no triage threshold")
p, c = fitpred(LogisticRegression(max_iter=3000, C=1), feats())
add("8a. LR C=1", "dropped", p, c, "")
p, c = fitpred(LogisticRegression(max_iter=3000, C=20), feats())
add("8b. LR C=20", "dropped", p, c, "")
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats(char=True))
add("9. LR + character n-grams", "dropped", p, c, "+0.7pp is inside the noise (+/-0.8pp on 2,135 rows); slower, bigger model, no real gain")
# ceiling: look the exact text up in train; majority label of identical texts
key = fit.groupby(["txt", "product_family"]).final.agg(lambda s: s.value_counts().idxmax())
look = ho.set_index(["txt", "product_family"]).index.map(lambda k: key.get(k, np.nan))
hit = pd.Series(look, index=ho.index).notna()
ceil = np.where(hit, pd.Series(look, index=ho.index).fillna(""), maj)
add("10. exact-text lookup (memorise train)", "dropped", ceil, None, "only %.0f%% of hold-out texts were seen before (order numbers, wording): memorising does not work" % (100 * hit.mean()))
add("11. LLM zero-shot / few-shot (hosted API)", "NOT RUN", None, None,
    "no per-request spend (finance condition); remaining errors are texts with no routing information, which a bigger model cannot recover")

# evidence for row 11: on texts the model finds vague, the best a memoriser can do ~ chance
p, c = fitpred(LogisticRegression(max_iter=3000, C=5), feats())
low = c < 0.54
vague_best = (ceil[low] == ho.final.values[low]).mean()
rows.append({"variant": "evidence: on the %.0f%% lowest-confidence texts, model / bot / memoriser accuracy" % (100 * low.mean()), "kept_or_dropped": "",
             "acc_vs_final": round(float((p[low] == ho.final.values[low]).mean()), 3), "acc_vs_bot": round(float((ho.bot.values[low] == ho.final.values[low]).mean()), 3),
             "acc_at_85pct_coverage": round(float(vague_best), 3), "why": "all three near chance (1 in 7 = 0.14): the words are not there"})
out = pd.DataFrame(rows); Path("out").mkdir(exist_ok=True); out.to_csv("out/experiments.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70)
print(out[["variant", "kept_or_dropped", "acc_vs_final", "acc_vs_bot", "acc_at_85pct_coverage"]].to_string(index=False))

# vague templates: final-team spread among train rows with the same opening phrase
pats = {"someone contact me about": r"someone contact me about", "call back regarding / call me about": r"please call back regarding|call me about",
        "complaint about": r"complaint about", "need help with my": r"need help with my", "service request for": r"service request for", "<product> problem": r"\bproblem\b"}
vr = []
for k, pt in pats.items():
    x = m[m.txt.str.contains(pt, regex=True)]; d = x.final.value_counts(normalize=True)
    vr.append({"opening_phrase": k, "n_train": len(x), "most_common_team": d.index[0], "its_share": round(d.iloc[0], 2),
               "teams_with_over_10pct": int((d > .10).sum()), "bot_correct": round((x.bot == x.final).mean(), 2)})
pd.DataFrame(vr).to_csv("out/vague_templates.csv", index=False)
print(); print(pd.DataFrame(vr).to_string(index=False))
