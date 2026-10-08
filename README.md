# Kestrel Home: service-request routing classifier

Predicts which team should get a service request, trained on **where requests actually ended up**
(not on the old bot's queue), and flags requests whose text is too vague to route.

No API keys. No network. No per-request cost. About 6 seconds for the full 18-month history.

## Run it (clean machine)

Needs Python 3.10+. **Mac/Linux**

```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data      # copy train.csv test_unlabelled.csv resolution_log.csv (and teams.csv, sample_submission.csv) in
python route.py
python experiments.py      # optional: the model-choice log
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir data
python route.py
python experiments.py
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

> **Data is not in this repo.** Kestrel's ops policy (s10) forbids publishing customer or operational
> data to public repositories. `data/` and `out/` (which contain request text) are git-ignored.

## Web UI (optional)

```bash
streamlit run app.py        # opens http://localhost:8501
```

Three tabs: **Route one request** (type a message, get the team, confidence, or an "ask the customer" flag),
**Route a CSV** (upload or use `test_unlabelled.csv`, download predictions), **Results** (bot vs model, money, busiest teams, model choices).
Needs `data/` filled in. Runs locally; nothing leaves the machine.

## Outputs in `out/`

| File | What |
|---|---|
| `submission.csv` | `request_id,team` for every row of `test_unlabelled.csv`, same format as `sample_submission.csv` |
| `predictions_detail.csv` | adds confidence, second choice and `needs_triage` |
| `metrics.json` | every number quoted in the memo |
| `per_team_holdout.csv`, `confusion_holdout.csv` | hold-out scores by team |
| `experiments.csv` | every model variant tried, scored the same way, with kept/dropped and why |
| `vague_templates.csv` | evidence that vague opening phrases end up in many different teams |
| `workload_by_team.csv`, `workload_forecast_jul_sep_2026.csv` | busiest teams, by where requests actually close |
| `bot_misroute_matrix.csv` | which bot queue sends what to which team |

## What it does

1. **Merges renamed teams** (Installations -> Installs & Demo, Consumables -> Filters & Consumables, ops-policy s5).
2. **Repairs Zoho text** (`thÃ©`, `Ã¢â‚¬Â¦` -> plain text) and strips order/reference numbers.
3. **Fixes Zoho resolve times** (stored in UTC, never converted, s9): +5:30 to IST. Only used for the workload table.
4. **Trains on `final_team`** from `resolution_log.csv`. TF-IDF word 1-2 grams + channel + product + warranty, logistic regression.
5. **Scores on a time-ordered hold-out.** Train Apr-Dec 2025, choose the confidence threshold on Jan-Mar 2026, retrain on both, score Apr-Jun 2026 untouched.
6. **Flags low-confidence requests** (`needs_triage`) instead of guessing.
7. **Prices it** with the policy's costs (Rs 305 per transfer, Rs 260 per extra contact, Rs 3.2 lakh licence).

## Decisions the brief did not make

- **Ground truth = final team, not the bot's queue.** The client asked for 90% agreement with `team_label`. That is the bot's own output; a copy of it scores 94.8% against the bot but only 75.7% against reality.
- **Threshold 0.54** = lowest confidence at which auto-routed requests were at least 95% right on the validation window. On the hold-out it gave 94.3%.
- **Cost of a model error** = average cost of a bot misroute today (1.45 transfers x Rs 305 + Rs 260 = Rs 701).
- **Conservative money figure** treats every triage request as still misrouted. The optimistic figure assumes one Rs 260 clarifying contact fixes each.

## Layout

```
route.py         the tool
app.py           Streamlit UI (streamlit run app.py)
experiments.py   model-choice log
requirements.txt
```
