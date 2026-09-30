"""Small orchestrator demonstrating the Interviewer -> Evaluator handoff."""

import json
from typing import Any, Dict

from agent import EvaluatorAgent
from interviewer_agent import InterviewerAgent


class MultiAgentInterview:
    """Coordinate question selection and answer evaluation for one session."""

    def __init__(self):
        self.interviewer = InterviewerAgent()
        self.evaluator = EvaluatorAgent()

    def next_question(self) -> Dict[str, Any]:
        return self.interviewer.choose_next_question(self.evaluator.history)

    def submit_answer(self, question: Dict[str, Any], answer: str) -> Dict[str, Any]:
        if not question.get("question"):
            raise ValueError("There are no unanswered questions left.")
        return self.evaluator.evaluate(
            question["question"],
            answer,
            expected_keywords=question.get("keywords", []),
        )


def main() -> None:
    session = MultiAgentInterview()
    print("InterviewIQ multi-agent demo. Press Enter on an empty answer to stop.\n")

    while True:
        question = session.next_question()
        if not question["question"]:
            break

        print(f"Interviewer ({question['category']}): {question['question']}")
        print(f"Selection reason: {question['reason']}")
        answer = input("Your answer: ").strip()
        if not answer:
            break

        record = session.submit_answer(question, answer)
        print(f"Evaluator: {record['feedback']}\n")

    print(json.dumps(session.evaluator.generate_final_report(), indent=2))


if __name__ == "__main__":
    main()