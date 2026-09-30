# InterviewIQ — Problem Statement & Milestones

## The problem
Hiring teams often want a fast, first-pass read on a candidate's interview answers before a human reviews them. Your task is to build **InterviewIQ**: an AI agent that acts as a mock-interview coach with a working browser interface. Given a candidate's spoken answer to an interview question, the agent evaluates it using a set of tools, produces short feedback, remembers the whole session well enough to answer questions about it at any point, and surfaces all of that through a UI — not just a terminal.

## The asks

**Ask 1 — Build a tool-calling evaluator agent.**
Implement three rule-based tools the agent can call, each returning a structured dict:
1. `detect_filler_words` — flags filler words like "um", "like", "basically".
2. `check_star_structure` — checks whether a behavioral answer covers Situation, Task, Action, Result.
3. `score_relevance` — scores (0–100) how many expected keywords/concepts for the question appear in the answer.

The agent should read the candidate's answer, decide which tool(s) are relevant, call them, and turn the results into short, encouraging feedback.

**Ask 2 — Session memory with a real aggregation, verified by a memory check.**
The agent must remember every answer's evaluation across the session — not just the current one — and be able to answer meta-questions at *any point* mid-session, such as "how am I doing so far?" or "what's my weakest area?", not only produce a report at the very end. The report must do real aggregation: an average relevance score and a named **weakest area** (the question with the lowest score), not a flat recap of everything scored so far.

Run `memory_check.py` against your finished agent: it scores one strong answer and one deliberately weak one, then asks the meta-question. If your agent correctly names the weak answer's question as the weakest area — not whichever question was asked most recently — your memory and aggregation are working. If it gets this wrong, your agent is only looking at the last turn.

**Ask 3 — A working Gradio interface (required).**
Wrap the agent in a Gradio app (`app.py`) with: the current question and an answer box, a feedback panel, a live scorecard showing every question answered so far, a free-form box to ask the coach meta-questions, and a button to pull the final report. This is the interface you'll demo — not optional, and not just a CLI print loop.

**Ask 4 (bonus, not graded) — a second agent.**
None of the six classes actually built a multi-agent system — Class 6's guide only introduces the *vocabulary* (an "orchestrator" coordinating "specialized agents") and explicitly scopes out building one as "a larger topic for later." So this ask sits outside the core project on purpose: `interviewer_agent.py` and `run_multi_agent.py` are included **fully worked, not as a TODO** — a second agent (the "Interviewer") whose only job is to decide the next question's category based on how the candidate has done so far, handed off to the Evaluator agent from Asks 1–3 (completely unchanged) by a small orchestrator loop. Run `python run_multi_agent.py` to try it. Read it, run it, and extend it if you want to go further — there's nothing here you're required to build or that gets graded, since it goes beyond what was actually taught.

## Suggested approach
1. Get the plumbing working first (API key, one plain LLM call) before writing any tool logic.
2. Write and test each tool as a standalone Python function — don't wire it to the agent until you're confident it works on its own.
3. Wire the tools to the agent and get one full evaluated turn working via `main.py` before touching memory.
4. Add memory and the aggregation report, then run `memory_check.py` to prove it actually uses more than the last turn.
5. Only once `main.py` and `memory_check.py` both work should you move to `app.py` — the UI should be the last layer on top of an agent you already trust.

## Milestones
1. Environment setup — venv, `pip install -r requirements.txt`, `.env` configured with your API key.
2. First plain LLM call working (no tools yet) — sanity check that your key and provider are correct.
3. `detect_filler_words` implemented and tested directly.
4. `check_star_structure` implemented and tested directly.
5. `score_relevance` implemented and tested directly.
6. All three tools wired into the agent's tool-calling loop; one full evaluated turn works end-to-end via `main.py`.
7. Session memory added — running a few questions shows it accumulating.
8. `generate_final_report` implemented with real aggregation (average relevance + named weakest area).
9. `ask_agent` implemented so meta-questions work mid-session, not just at the end.
10. `python memory_check.py` passes — the weakest area named matches the deliberately weak answer, not the most recent turn.
11. `app.py` built — the full flow (question → answer → feedback → scorecard → meta-question → final report) works in the browser.
12. (Bonus, ungraded) Run `python run_multi_agent.py` and read `interviewer_agent.py` -- a second agent handed off to your Evaluator by a small orchestrator.

## About the helper files
- **`.env` / `.env.example`** — holds your API key(s) (e.g. `GROQ_API_KEY`).
- **`.gitignore`** — tells git to ignore your virtual environment.
- **`requirements.txt`** — the exact list of Python packages this project needs (`openai`, `python-dotenv`, `gradio`).

## What "successful completion" looks like
Running `python app.py` should let you go through every question in `interview_bank.py` in the browser, see per-answer feedback and a live scorecard, ask the coach a meta-question mid-session and get an answer that correctly reflects the *whole* session, and pull a final report naming the weakest area. `python memory_check.py` should pass.
