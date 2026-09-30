"""Specialized agent that chooses the next interview question."""

from typing import Any, Dict, List, Optional


QUESTION_BANK: List[Dict[str, Any]] = [
    {
        "category": "behavioral",
        "question": "Tell me about a time you solved a difficult problem at work.",
        "keywords": ["problem", "root cause", "action", "result", "improved"],
    },
    {
        "category": "behavioral",
        "question": "Tell me about a time you had to adapt to a major change.",
        "keywords": ["change", "adapted", "challenge", "action", "result"],
    },
    {
        "category": "technical",
        "question": "What experience do you have with Python?",
        "keywords": ["Python", "project", "automation", "API", "data"],
    },
    {
        "category": "technical",
        "question": "How would you troubleshoot a slow Python application?",
        "keywords": ["Python", "profiling", "performance", "database", "optimization"],
    },
    {
        "category": "communication",
        "question": "Tell me about a time you resolved a conflict on your team.",
        "keywords": ["conflict", "team", "communication", "listened", "resolution"],
    },
    {
        "category": "communication",
        "question": "How do you explain a complex technical idea to a non-technical audience?",
        "keywords": ["explain", "technical", "audience", "simple", "communication"],
    },
]


class InterviewerAgent:
    """Choose questions while leaving answer evaluation to EvaluatorAgent."""

    def __init__(self, question_bank: Optional[List[Dict[str, Any]]] = None):
        self.question_bank = question_bank or QUESTION_BANK

    def choose_next_question(self, evaluation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the unanswered question from the weakest covered category."""
        answered = {record.get("question") for record in evaluation_history}
        remaining = [
            item for item in self.question_bank if item["question"] not in answered
        ]
        if not remaining:
            return {
                "category": None,
                "question": None,
                "keywords": [],
                "reason": "All configured questions have been answered.",
            }

        category_scores: Dict[str, List[int]] = {}
        for record in evaluation_history:
            category = self._category_for(record.get("question"))
            if category is None:
                continue
            score = record.get("results", {}).get("score_relevance", {}).get("score", 0)
            category_scores.setdefault(category, []).append(score)

        remaining_categories = {item["category"] for item in remaining}
        scored_remaining_categories = remaining_categories.intersection(category_scores)

        if not scored_remaining_categories:
            selected = remaining[0]
            if not category_scores:
                reason = "Starting with a balanced behavioral question."
            else:
                reason = (
                    f"No unanswered question remains in the previously scored categories; "
                    f"moving to {selected['category']}."
                )
        else:
            weakest_category = min(
                scored_remaining_categories,
                key=lambda category: sum(category_scores[category]) / len(category_scores[category]),
            )
            selected = next(
                (item for item in remaining if item["category"] == weakest_category),
                remaining[0],
            )
            average = sum(category_scores[weakest_category]) / len(category_scores[weakest_category])
            reason = (
                f"The {weakest_category} category is currently weakest "
                f"({average:.0f}/100 average relevance)."
            )

        return {**selected, "reason": reason}

    def _category_for(self, question: Optional[str]) -> Optional[str]:
        for item in self.question_bank:
            if item["question"] == question:
                return item["category"]
        return None