import html
import json
from typing import Any, Dict, List, Tuple

import gradio as gr

from agent import EvaluatorAgent


QUESTION_BANK: List[Dict[str, Any]] = [
    {
        "question": "Tell me about a time you solved a difficult problem at work.",
        "keywords": ["problem", "root cause", "action", "result", "improved"],
    },
    {
        "question": "What experience do you have with Python?",
        "keywords": ["Python", "project", "automation", "API", "data"],
    },
    {
        "question": "Tell me about a time you resolved a conflict on your team.",
        "keywords": ["conflict", "team", "communication", "listened", "resolution"],
    },
]


def question_options() -> List[str]:
    return [item["question"] for item in QUESTION_BANK]


def keywords_for(question: str) -> List[str]:
    for item in QUESTION_BANK:
        if item["question"] == question:
            return item["keywords"]
    return []


def decode_history(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def format_feedback(record: Dict[str, Any]) -> str:
    relevance = record["results"].get("score_relevance", {})
    star = record["results"].get("check_star_structure", {})
    filler = record["results"].get("detect_filler_words", {})

    lines = [
        "### Latest feedback",
        record["llm_feedback"] or record["feedback"],
        "",
        f"**Relevance:** {relevance.get('score', 0)}/100",
    ]
    if star:
        lines.append(
            f"**STAR coverage:** {', '.join(star.get('covered', [])) or 'None'} "
            f"({star.get('score', 0)}/100)"
        )
    lines.append(f"**Filler words:** {filler.get('count', 0)}")
    return "\n".join(lines)


def format_scorecard(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No answers evaluated yet. Your completed answers will appear here."

    rows = [
        "| # | Question | Relevance | STAR | Fillers |",
        "|---:|---|---:|---:|---:|",
    ]
    for index, record in enumerate(history, start=1):
        question = html.escape(record["question"].replace("|", "\\|"))
        relevance = record["results"].get("score_relevance", {}).get("score", 0)
        star = record["results"].get("check_star_structure", {}).get("score", "-")
        fillers = record["results"].get("detect_filler_words", {}).get("count", 0)
        rows.append(f"| {index} | {question} | {relevance}/100 | {star}/100 | {fillers} |")
    return "\n".join(rows)


def submit_answer(
    question: str,
    answer: str,
    history: List[Dict[str, Any]],
) -> Tuple[str, str, str]:
    agent = EvaluatorAgent()
    agent.history = decode_history(history)

    if not question:
        return "Please select a question first.", format_scorecard(agent.history), json.dumps(agent.history)
    if not answer or not answer.strip():
        return "Please enter an answer before submitting.", format_scorecard(agent.history), json.dumps(agent.history)

    record = agent.evaluate(question, answer, expected_keywords=keywords_for(question))
    return format_feedback(record), format_scorecard(agent.history), json.dumps(agent.history)


def ask_coach(coach_question: str, history: List[Dict[str, Any]]) -> str:
    if not coach_question or not coach_question.strip():
        return "Type a question about your progress first."
    agent = EvaluatorAgent()
    agent.history = decode_history(history)
    return agent.ask_agent(coach_question)


def final_report(history: List[Dict[str, Any]]) -> str:
    agent = EvaluatorAgent()
    agent.history = decode_history(history)
    report = agent.generate_final_report()
    if not report["evaluations"]:
        return "### Final report\nNo answers have been evaluated yet."

    weakest = report["weakest_area"]
    return (
        "### Final report\n"
        f"**Answers evaluated:** {report['evaluations']}\n\n"
        f"**Average relevance:** {report['average_relevance']}/100\n\n"
        f"**Weakest area:** {weakest['question']} ({weakest['score']}/100)\n\n"
        f"{report['summary']}"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="InterviewIQ Coach") as app:
        # Keep session history in a normal component so Gradio can serialize it reliably.
        agent_state = gr.Textbox(value="[]", visible=False, label="session history")

        gr.Markdown(
            "# InterviewIQ Coach\n"
            "Practice an answer, get focused feedback, and track your progress across the session."
        )

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Dropdown(
                    choices=question_options(),
                    value=QUESTION_BANK[0]["question"],
                    label="Current question",
                )
                answer = gr.Textbox(
                    label="Your answer",
                    placeholder="Write or paste your interview answer here...",
                    lines=10,
                )
                submit = gr.Button("Evaluate answer", variant="primary")

            with gr.Column(scale=1):
                feedback = gr.Markdown("Submit an answer to receive feedback.")
                scorecard = gr.Markdown(format_scorecard([]))

        gr.Markdown("## Ask your coach")
        with gr.Row():
            coach_question = gr.Textbox(
                label="Meta-question",
                placeholder="How am I doing so far? What's my weakest area?",
                scale=4,
            )
            ask_button = gr.Button("Ask coach", scale=1)
        coach_response = gr.Markdown()

        report_button = gr.Button("Generate final report")
        report = gr.Markdown()

        submit.click(
            submit_answer,
            inputs=[question, answer, agent_state],
            outputs=[feedback, scorecard, agent_state],
        )
        ask_button.click(
            ask_coach,
            inputs=[coach_question, agent_state],
            outputs=coach_response,
        )
        report_button.click(
            final_report,
            inputs=agent_state,
            outputs=report,
        )

    return app


app = build_app()


if __name__ == "__main__":
    app.launch(show_error=True)
