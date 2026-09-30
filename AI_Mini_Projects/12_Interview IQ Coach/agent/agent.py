import re
from typing import Any, Dict, List, Optional

from tools import (
    call_groq_for_feedback,
    check_star_structure,
    decide_relevant_tools,
    detect_filler_words,
    score_relevance,
)


class EvaluatorAgent:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def generate_final_report(self) -> Dict[str, Any]:
        """Aggregate every evaluation recorded in this session."""
        if not self.history:
            return {
                "evaluations": 0,
                "average_relevance": 0,
                "weakest_area": None,
                "summary": "No answers have been evaluated yet.",
            }

        relevance_scores = [
            record["results"].get("score_relevance", {}).get("score", 0)
            for record in self.history
        ]
        weakest_record = min(
            self.history,
            key=lambda record: record["results"].get("score_relevance", {}).get("score", 0),
        )
        weakest_score = weakest_record["results"].get("score_relevance", {}).get("score", 0)
        average_relevance = round(sum(relevance_scores) / len(relevance_scores), 2)

        return {
            "evaluations": len(self.history),
            "average_relevance": average_relevance,
            "weakest_area": {
                "question": weakest_record["question"],
                "score": weakest_score,
            },
            "summary": (
                f"You completed {len(self.history)} evaluation(s) with an average relevance "
                f"score of {average_relevance}/100. Your weakest area is the question "
                f"'{weakest_record['question']}' at {weakest_score}/100."
            ),
        }

    def ask_agent(self, question: str) -> str:
        """Answer session-level questions using the complete evaluation history."""
        if not self.history:
            return "I do not have any evaluated answers yet. Submit an answer first."

        report = self.generate_final_report()
        normalized_question = question.lower()

        if re.search(r"weakest|weak area|improve most|needs work", normalized_question):
            weakest = report["weakest_area"]
            return (
                f"Your weakest area so far is '{weakest['question']}', with a relevance "
                f"score of {weakest['score']}/100. Focus on directly addressing the key concepts in that question."
            )

        if re.search(r"how am i doing|how have i done|progress|so far|overall", normalized_question):
            return report["summary"] + " Keep going: each answer gives you a concrete place to improve."

        return (
            f"So far I have evaluated {report['evaluations']} answer(s). Your average relevance "
            f"score is {report['average_relevance']}/100."
        )

    def evaluate(self, question: str, answer: str, expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        tools_used = decide_relevant_tools(question, answer)
        results: Dict[str, Any] = {}

        for tool_name in tools_used:
            if tool_name == "detect_filler_words":
                results[tool_name] = detect_filler_words(answer)
            elif tool_name == "check_star_structure":
                results[tool_name] = check_star_structure(answer)
            elif tool_name == "score_relevance":
                results[tool_name] = score_relevance(
                    answer,
                    expected_keywords=expected_keywords,
                    question=question,
                )

        feedback_parts = []
        filler = results.get("detect_filler_words", {})
        star = results.get("check_star_structure", {})
        relevance = results.get("score_relevance", {})

        if filler.get("count", 0) >= 3:
            feedback_parts.append("You have a few filler words, so try pausing before you answer to sound more confident.")
        elif filler.get("count", 0):
            feedback_parts.append("Your answer is mostly clear; a couple of filler phrases can be trimmed for polish.")

        missing = star.get("missing", [])
        if missing:
            feedback_parts.append(
                f"A stronger story would include more detail on {', '.join(missing)} to make the example easier to follow."
            )

        score = relevance.get("score", 100)
        if score < 70:
            feedback_parts.append("You are close. Add more of the key ideas from the question to show stronger alignment.")
        elif score >= 80:
            feedback_parts.append("Nice job tying your answer back to the key points in the question.")

        if not feedback_parts:
            feedback_parts.append("Strong answer overall—you communicated your point clearly and confidently.")

        summary = " ".join(feedback_parts)

        llm_feedback = None
        if "GROQ_API_KEY" in __import__("os").environ:
            llm_feedback = call_groq_for_feedback(
                "You help interview candidates by giving short, encouraging and specific feedback.",
                f"Question: {question}\nAnswer: {answer}\nTool results: {results}",
            )

        record = {
            "question": question,
            "answer": answer,
            "tools_used": tools_used,
            "results": results,
            "feedback": summary,
            "llm_feedback": llm_feedback,
        }
        self.history.append(record)
        return record


if __name__ == "__main__":
    agent = EvaluatorAgent()
    result = agent.evaluate(
        "Tell me about a time you solved a difficult problem at work.",
        "Um, I was working on a project and we had a big issue. I needed to fix it quickly. I organized the team, looked at the root cause, and built a plan. As a result, we reduced errors by 30 percent and shipped on time.",
        expected_keywords=["project", "root cause", "team", "reduce errors", "ship on time"],
    )
    print(result)
