"""Kestrel routing UI.   Run:  streamlit run app.py
Three tabs: try one request, route a whole CSV, see the results behind the memo.
Everything runs locally; nothing is sent anywhere."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import route

DATA, OUT = Path("data"), Path("out")
st.set_page_config(page_title="Kestrel routing", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Training on your history (a few seconds)...")
def get_model():
    m, te = route.load(DATA)
    A = m[m.created < route.VAL_START]; B = m[(m.created >= route.VAL_START) & (m.created < route.TEST_START)]
    mdl = route.make_model().fit(A, A.final)
    pB, cB, _ = route.predict(mdl, B)
    thr = route.pick_threshold(cB, pB == B.final.values)
    full = route.make_model().fit(m, m.final)
    return full, thr, m


def classify(df, full, thr):
    df = df.copy()
    df["txt"] = df.request_text.map(route.fix_text)
    pred, conf, second = route.predict(full, df)
    return pd.DataFrame({"request_id": df.get("request_id", pd.Series(range(len(df)), index=df.index)),
                         "team": pred, "confidence": conf.round(3), "second_choice": second,
                         "action": np.where(conf >= thr, "Auto-route", "Ask customer / triage")})


st.title("🧭 Kestrel service-request routing")
st.caption("Trained on the team that actually closed each request, not on the old bot's queue.")

if not all((DATA / f).exists() for f in ["train.csv", "test_unlabelled.csv", "resolution_log.csv"]):
    st.error("Put train.csv, test_unlabelled.csv and resolution_log.csv in the data/ folder, then reload.")
    st.stop()

full, thr, hist = get_model()
t1, t2, t3 = st.tabs(["Route one request", "Route a CSV", "Results"])

with t1:
    c1, c2 = st.columns([3, 2])
    with c1:
        text = st.text_area("Customer message", "water purifier stopped working after 9 months paid on upi", height=110)
    with c2:
        channel = st.selectbox("Channel", ["chat", "whatsapp", "ivr", "email"])
        product = st.selectbox("Product", sorted(hist.product_family.unique()))
        warranty = st.selectbox("Warranty status", ["in_warranty", "shield", "out_of_warranty"])
    if st.button("Route it", type="primary") and text.strip():
        r = classify(pd.DataFrame([{"request_text": text, "channel": channel, "product_family": product, "warranty_status": warranty}]), full, thr).iloc[0]
        if r.action == "Auto-route":
            st.success(f"**{r.team}**  ·  confidence {r.confidence:.0%}")
        else:
            st.warning(f"Too vague to route safely. Best guess **{r.team}** ({r.confidence:.0%}), then **{r.second_choice}**. Ask the customer: fault, delivery, payment, or how-to?")
        old = {"payment": "Billing", "paid": "Billing"}
        if any(w in text.lower() for w in old):
            st.info("Note: the old bot sends anything mentioning payment to Billing. Billing is right only when the payment itself is the problem.")

with t2:
    st.write("Upload a CSV with columns `request_id, channel, product_family, warranty_status, request_text` (same as test_unlabelled.csv).")
    up = st.file_uploader("CSV file", type="csv")
    use_test = st.checkbox("Or use the provided test_unlabelled.csv", value=up is None)
    src = pd.read_csv(up) if up is not None else (pd.read_csv(DATA / "test_unlabelled.csv") if use_test else None)
    if src is not None:
        need = {"channel", "product_family", "warranty_status", "request_text"}
        if not need <= set(src.columns):
            st.error(f"Missing columns: {sorted(need - set(src.columns))}")
        else:
            res = classify(src, full, thr)
            a, b, c = st.columns(3)
            a.metric("Requests", f"{len(res):,}")
            b.metric("Auto-routed", f"{(res.action == 'Auto-route').mean():.0%}")
            c.metric("Need a question first", f"{(res.action != 'Auto-route').mean():.0%}")
            st.bar_chart(res.team.value_counts())
            st.dataframe(res, width="stretch", height=300)
            st.download_button("Download predictions (CSV)", res.to_csv(index=False), "predictions.csv", "text/csv")
            st.download_button("Download in submission format (request_id, team)", res[["request_id", "team"]].to_csv(index=False), "submission.csv", "text/csv")

with t3:
    if not (OUT / "metrics.json").exists():
        if st.button("Run the full evaluation now (about 10 seconds)"):
            subprocess.run([sys.executable, "route.py"], check=True); st.rerun()
        st.info("Run `python route.py` once to fill this tab.")
    else:
        R = json.load(open(OUT / "metrics.json"))
        st.subheader("Is it better than the bot?  (hold-out: Apr-Jun 2026, never seen in training)")
        k = st.columns(4)
        k[0].metric("Bot correct", f"{R['bot_acc_holdout']:.1%}")
        k[1].metric("Model correct", f"{R['model_acc_all']:.1%}", f"{(R['model_acc_all'] - R['bot_acc_holdout']) * 100:+.1f} pts")
        k[2].metric("Model on confident 85%", f"{R['model_acc_when_routed']:.1%}", f"bot {R['bot_acc_on_same_routed']:.1%}", delta_color="off")
        k[3].metric("Needs a question", f"{1 - R['auto_route_share']:.0%}")
        st.warning(f"The 90% brief: a copy of the bot matches the bot {R['bot_clone_matches_bot']:.1%} but the real final team only {R['bot_clone_matches_final']:.1%}.")
        st.subheader("Money (policy costs, hold-out scaled to a year)")
        m_ = st.columns(3)
        m_[0].metric("Transfers needed: today", f"{R['transferred_share_bot']:.1%}")
        m_[1].metric("Conservative", f"{R['transferred_share_model_conservative']:.1%}", f"Rs {R['annual_total_inr'] / 1e5:.1f} lakh/yr", delta_color="off")
        m_[2].metric("With triage question", f"{R['transferred_share_model_if_triage_fixed']:.1%}", f"Rs {R['annual_saving_if_triage_fixed_inr'] / 1e5:.1f} lakh/yr", delta_color="off")
        st.caption("Savings include the Rs 3.2 lakh bot licence. Run cost: Rs 0, no paid API.")
        l, r_ = st.columns(2)
        with l:
            st.subheader("Busiest teams (where requests actually close)")
            w = pd.read_csv(OUT / "workload_by_team.csv").set_index("final")
            st.bar_chart(w.requests_per_month)
            st.dataframe(w[["requests_per_month", "share_pct", "bot_queue_vs_actual_pct"]].rename(columns={"bot_queue_vs_actual_pct": "bot queue vs actual %"}))
        with r_:
            st.subheader("Per-team hold-out score")
            st.dataframe(pd.read_csv(OUT / "per_team_holdout.csv").set_index("team"))
            st.subheader("Model choices tried")
            if (OUT / "experiments.csv").exists():
                st.dataframe(pd.read_csv(OUT / "experiments.csv")[["variant", "kept_or_dropped", "acc_vs_final"]], height=250)
