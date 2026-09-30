"""Gradio interface for the Interviewer -> Evaluator multi-agent session."""

import html
import json
from typing import Any, Dict, List, Tuple

import gradio as gr

from interviewer_agent import InterviewerAgent
from run_multi_agent import MultiAgentInterview


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


def question_details(question: Dict[str, Any]) -> str:
    if not question.get("question"):
        return "### Interviewer\nAll configured questions have been answered."
    return (
        f"### Interviewer ({question['category'].title()})\n"
        f"{question['question']}\n\n"
        f"_{question['reason']}_"
    )


def format_feedback(record: Dict[str, Any]) -> str:
    results = record["results"]
    relevance = results.get("score_relevance", {})
    star = results.get("check_star_structure", {})
    filler = results.get("detect_filler_words", {})
    return "\n".join(
        [
            "### Evaluator feedback",
            record["llm_feedback"] or record["feedback"],
            "",
            f"**Relevance:** {relevance.get('score', 0)}/100",
            f"**STAR coverage:** {', '.join(star.get('covered', [])) or 'None'} "
            f"({star.get('score', 0)}/100)",
            f"**Filler words:** {filler.get('count', 0)}",
        ]
    )


def format_scorecard(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No answers evaluated yet."

    rows = [
        "| # | Question | Relevance | STAR | Fillers |",
        "|---:|---|---:|---:|---:|",
    ]
    for index, record in enumerate(history, start=1):
        question = html.escape(record["question"].replace("|", "\\|"))
        results = record.get("results", {})
        relevance = results.get("score_relevance", {}).get("score", 0)
        star = results.get("check_star_structure", {}).get("score", "-")
        fillers = results.get("detect_filler_words", {}).get("count", 0)
        rows.append(f"| {index} | {question} | {relevance}/100 | {star}/100 | {fillers} |")
    return "\n".join(rows)


def initial_question() -> Tuple[str, str]:
    question = InterviewerAgent().choose_next_question([])
    return question_details(question), json.dumps(question)


def submit_answer(
    question_json: str,
    answer: str,
    history_json: str,
) -> Tuple[str, str, str, str, str, str]:
    history = decode_history(history_json)
    try:
        question = json.loads(question_json)
    except (TypeError, json.JSONDecodeError):
        question = {}

    if not question.get("question"):
        return (
            "There are no unanswered questions left.",
            question_details(question),
            format_scorecard(history),
            history_json or "[]",
            question_json or "{}",
            answer,
        )
    if not answer or not answer.strip():
        return (
            "Please enter an answer before submitting.",
            question_details(question),
            format_scorecard(history),
            history_json or "[]",
            question_json,
            answer,
        )

    session = MultiAgentInterview()
    session.evaluator.history = history
    record = session.submit_answer(question, answer)
    next_question = session.next_question()
    return (
        format_feedback(record),
        question_details(next_question),
        format_scorecard(session.evaluator.history),
        json.dumps(session.evaluator.history),
        json.dumps(next_question),
        "",
    )


def final_report(history_json: str) -> str:
    session = MultiAgentInterview()
    session.evaluator.history = decode_history(history_json)
    report = session.evaluator.generate_final_report()
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
    question_json = initial_question()[1]
    with gr.Blocks(title="InterviewIQ Multi-Agent Coach") as app:
        history_state = gr.Textbox(value="[]", visible=False, label="session history")
        current_question_state = gr.Textbox(
            value=question_json,
            visible=False,
            label="current question",
        )

        gr.Markdown(
            "# InterviewIQ Multi-Agent Coach\n"
            "The Interviewer chooses the next category; the Evaluator scores your answer."
        )
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Markdown()
                answer = gr.Textbox(
                    label="Your answer",
                    placeholder="Write or paste your interview answer here...",
                    lines=10,
                )
                submit = gr.Button("Submit answer", variant="primary")
            with gr.Column(scale=1):
                feedback = gr.Markdown("Submit an answer to receive evaluator feedback.")
                scorecard = gr.Markdown(format_scorecard([]))

        gr.Markdown("## Session report")
        report_button = gr.Button("Generate final report")
        report = gr.Markdown()

        app.load(
            lambda: question_details(json.loads(question_json)),
            outputs=question,
        )
        submit.click(
            submit_answer,
            inputs=[current_question_state, answer, history_state],
            outputs=[
                feedback,
                question,
                scorecard,
                history_state,
                current_question_state,
                answer,
            ],
        )
        report_button.click(final_report, inputs=history_state, outputs=report)

    return app


app = build_app()


if __name__ == "__main__":
    app.launch(show_error=True)