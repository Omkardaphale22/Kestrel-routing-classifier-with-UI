# 3-minute screen recording: script (you record it; keep it under 3:00)

Screen: terminal + out/experiments.csv + this chat. Voice on. No slides.

**0:00-0:25  Run it.** `python route.py`. Point at the first two lines: bot correct 77.2% of the time; a clone of the bot scores 94.8% against the bot but 75.7% against reality. "The 90% bar is a trap."

**0:25-1:15  Prompts, in order.**
1. "Profile the CSVs": found renamed teams, mojibake, Zoho times in UTC, and that team_label equals the bot's first team on 100% of rows.
2. "Compare bot queue with final team": 25% of requests needed a transfer; errors cluster in Billing, Consumables, Repairs queues.
3. "Train on the bot's label vs the final team": the key experiment (open experiments.csv, rows 2 and 6).
4. "What do the errors look like?": vague texts, "someone contact me about X", spread over 5-7 teams (open vague_templates.csv).

**1:15-2:05  What changed between versions.**
- v1 trained on team_label as the client asked: 94.8% vs bot. I switched the target to final_team.
- v1 picked the threshold on the same data I reported. I added a separate validation window (Jan-Mar) and a hold-out (Apr-Jun).
- First money figure assumed triage was free. I made the headline conservative and kept the optimistic one separate.

**2:05-2:45  What I threw away.** Say it from experiments.csv: SVM and character n-grams (+0.1 to +0.7pp, inside the noise), Naive Bayes (worse), memorising texts (only a few percent seen before). The LLM was never run: no per-request spend, and the failures are missing information, not missing language skill.

**2:45-3:00  The honest part.** I could not score the July-September file (no answers). The 84% is from April-June. Repo contains code only because Kestrel's policy bans publishing the data.
