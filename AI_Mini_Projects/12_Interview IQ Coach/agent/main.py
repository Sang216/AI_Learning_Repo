import json

from agent import EvaluatorAgent


def main():
    question = "Tell me about a time you solved a difficult problem at work."
    answer = (
        "Um, I was working on a project and we had a big issue. I needed to fix it quickly. "
        "I organized the team, looked at the root cause, and built a plan. As a result, "
        "we reduced errors by 30 percent and shipped on time."
    )
    expected_keywords = ["project", "root cause", "team", "reduce errors", "ship on time"]

    agent = EvaluatorAgent()
    result = agent.evaluate(question, answer, expected_keywords=expected_keywords)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
