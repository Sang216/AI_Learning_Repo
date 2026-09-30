from agent import EvaluatorAgent


def main():
    agent = EvaluatorAgent()

    strong_question = "What experience do you have with Python?"
    strong_answer = "I have built Python APIs and automated data pipelines using Python at work."
    agent.evaluate(
        strong_question,
        strong_answer,
        expected_keywords=["Python", "APIs", "automated", "data pipelines"],
    )

    weak_question = "Tell me about a time you resolved a conflict on your team."
    weak_answer = "I like working with people."
    agent.evaluate(
        weak_question,
        weak_answer,
        expected_keywords=["conflict", "team", "communication", "resolution"],
    )

    response = agent.ask_agent("What's my weakest area so far?")
    report = agent.generate_final_report()
    expected_question = weak_question

    assert report["evaluations"] == 2
    assert report["average_relevance"] == 50
    assert report["weakest_area"]["question"] == expected_question
    assert expected_question in response

    print("PASS: session memory contains both evaluations")
    print(f"Average relevance: {report['average_relevance']}/100")
    print(f"Weakest area: {report['weakest_area']['question']}")
    print(f"Meta-question response: {response}")


if __name__ == "__main__":
    main()