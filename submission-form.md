# Submission form: Kestrel Home, service-request routing (Variant B)

> Items marked **[YOU]** are things only you can fill in truthfully.

**1. What did you build, and what business outcome does it move? State the number and the money.**
A routing classifier (`route.py`) that predicts the team that will actually close a request, flags requests too vague to route, and writes `submission.csv` for the 2,178 unlabelled requests. **Goal: cut requests that need a transfer from 25.0% to 19.8% now, worth about Rs 6.3 lakh a year (Rs 3.1 lakh in transfer and contact costs + the Rs 3.2 lakh bot licence), and to about 5% once a one-question triage step is live, worth about Rs 11.9 lakh a year.** Basis: Apr-Jun 2026 hold-out, policy costs (Rs 305 per transfer, Rs 260 per extra contact; a typical misroute costs Rs 701), scaled x4. The client's own target (90% match to the bot's labels) is met (94.8%) but would reproduce the bot's mistakes: a copy of the bot is right on only 75.7% of requests.

**2. What does one run cost, and what would a month cost at Kestrel's volume?**
No paid calls: Rs 0. The export holds 13,000 requests in 18 months, about 722 a month (about 167 a week). One full run (train on 10,822 rows, score 2,178) takes about 6 seconds on a laptop. A month of live traffic is about 722 predictions, which is a fraction of a second on the same machine: 722 x Rs 0 = Rs 0 usage cost. Hosting on Kestrel's own server is not costed, and I have not priced someone's time to retrain quarterly. I did not run or cost an LLM.

**3. How do you know it works?**
Time-ordered check, so the model never sees the future: train on Apr-Dec 2025 and choose the threshold on Jan-Mar 2026, then retrain on both and score Apr-Jun 2026 (2,135 requests, labels = final team).
- Bot: 76.7% right. Model: 84.0% right on everything, 94.3% right on the 85.1% it is confident about (bot on the same rows: 85.7%).
- On the 14.9% flagged vague, model and bot both score 25.2% (guessing among seven is 14%). About 70% of the model's errors sit in that group.
- Per team the model's F1 is 0.81-0.88 (`per_team_holdout.csv`).
- Margin of error is roughly +/-0.8 points on 2,135 rows, so differences under about 1 point between model variants are noise.
- The unlabelled July-September file cannot be scored. The submission has correct format (same ids, order, columns, valid team names, no old team names).
Where it goes wrong: messages with no routing information ("someone contact me about my purifier", "complaint about X"), and a few template collisions such as "replacement filter" with a damaged-box phrase.

**4. Did you change, narrow, or push back on the client's ask?** *(can only raise your score)*
- Ritu asked for a classifier matching the labels at 90%+. I pushed back: `team_label` is the bot's own queue and is wrong for 1 request in 4 (it equals the final team on 77.2%). I trained on `final_team` from the resolution log instead and said so in the memo's first bullet, with the number that shows why.
- Farhan wants the monthly run cost in writing: the memo states Rs 0 and why (no paid API).
- Ritu's side question on busiest teams: answered by where requests close, not by bot queue, because the queues are distorted by up to 37%.
- I did not promise to retire the bot outright. I recommended four weeks in shadow mode first.

**5. What is wrong with what you are handing us?** *(can only raise your score)*
- The July-September predictions are unscored. The 84% is the April-June hold-out; drift since is unknown.
- Money figures come from one 3-month hold-out scaled x4. The Rs 11.9 lakh case assumes one clarifying question fixes every vague request and costs only Rs 260; that is a hope, not a measurement. The headline Rs 6.3 lakh case is the conservative one.
- A model error is priced at the bot's average misroute (Rs 701), not measured.
- 477 requests (4.4%) have transfer counts that disagree with first/final team (351 with transfers but same team, 126 with none but different team). I used final team for the label and the transfer count for cost. I did not investigate further.
- Threshold 0.54 gave 94.3% on the hold-out, slightly under the 95% I aimed for on validation.
- Zoho UTC fix is applied only to the workload resolution times, which I did not use for any claim.
- Headcount view counts requests, not hours. Handle time is not in the data.
- The model reads text only through word counts. It cannot tell "wall mounting pending" (Installs) from "claim status" (Warranty) when both appear in one message.

**6. What did you deliberately leave out, and why that rather than something else?**
- An LLM or hosted API: Farhan ruled out per-request spend, and the remaining errors are missing information, not missing language skill (`vague_templates.csv`, `experiments.csv`).
- A live service or dashboard: Ritu needs a decision and a prediction file, not an app.
- Resolution-time and SLA analysis: not asked, and handle time is not recorded.
- Tuning further for the last 1-2 points: differences below 1 point are inside the noise.
- Feature work on order numbers, customer history or time of day: no evidence they carry routing information.

**7. Anything you built or found that nobody asked for?**
- Meenal's three observations are in the data: 35% of the Billing queue ends elsewhere, almost all with "paid" wording; 21.6% of the Consumables queue is Repairs, 98% of them purifiers; and the "call me about my purifier" pile is real and unroutable.
- The bot is about 94% right for Installs, Advice, Returns and Warranty queues, and about 60-65% right for Billing, Consumables and Repairs. That is where transfers come from.
- The bot's queue sizes overstate Billing by 26% and Consumables by 37% (headcount trap).
- Kestrel's policy forbids putting data in public repos, so the repo contains code only.

**8. What did you use AI for?** Link your recording: **[YOU: paste link]**
Claude (claude.ai chat, no API): profiling the data, writing `route.py` and `experiments.py`, drafting the memo and this form. It helped most at reading the misrouted texts quickly and at designing the time-ordered check. It wasted effort on a first version that trained on the bot's labels exactly as the brief said, which I threw away once the final-team column showed 75.7%. Model choices tested and dropped are in `out/experiments.csv`: always-Repairs, bot clone, raw text, text-only, Naive Bayes, SVM, two regularisation settings, character n-grams, exact-text lookup. **No LLM was run**, for the reasons in question 6. Cost: Rs 0 beyond **[YOU: your plan]**.

**9. Your Public Google Drive Link:** **[YOU: upload memo, recording, `submission.csv`; paste link]**
Note: `predictions_detail.csv` and the hold-out files contain customer request text. Under Kestrel's policy s10, share them only with the engagement team (private Drive link), not publicly.

**10. Someone picks this up on Monday and you are unreachable. The three things they need to know.**
1. Run `python route.py` (README). Everything in the memo is in `out/metrics.json`.
2. Do not use the 90%-match-to-bot target: use "closes in the first team it reaches". Run four weeks in shadow before switching.
3. The unclear 15% need a one-question triage step; no model fixes them. Retrain quarterly.

**11. Honest hours spent:** **[YOU: one number]**

**12. Github Repo Link:** **[YOU: paste URL]** (code only: `data/` and `out/` are git-ignored)
