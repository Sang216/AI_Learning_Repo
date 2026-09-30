import os
import re
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None


load_dotenv()

FILLER_WORDS = {
    "um": 1,
    "uh": 1,
    "like": 1,
    "basically": 1,
    "literally": 1,
    "actually": 1,
    "sort of": 1,
    "kind of": 1,
    "you know": 1,
    "i mean": 1,
    "honestly": 1,
    "obviously": 1,
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of",
    "in", "on", "at", "by", "with", "as", "from", "about", "into", "over", "under",
    "through", "between", "during", "after", "before", "while", "when", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "itself", "they", "them", "their", "he", "she", "i", "we", "you", "your",
    "our", "me", "my", "mine", "us", "his", "her", "hers", "themself", "themselves",
    "what", "which", "who", "how", "why", "where", "do", "does", "did", "have",
    "has", "had", "can", "could", "would", "should", "may", "might", "so", "very",
    "more", "most", "much", "many", "some", "any", "all", "not", "no", "yes", "just",
    "also", "too", "again", "there", "here", "than", "then", "because", "using",
    "used", "each", "few", "other", "same", "such", "own",
}


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _strip_stopwords(tokens: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        word = re.sub(r"[^a-z0-9]+", "", token.lower())
        if word and word not in STOP_WORDS and len(word) > 1:
            cleaned.append(word)
    return cleaned


def detect_filler_words(answer: str) -> Dict[str, Any]:
    """Detect common filler phrases and return a structured result."""
    text = _normalize_text(answer)
    if not text:
        return {
            "tool": "detect_filler_words",
            "status": "no_answer",
            "filler_words": [],
            "count": 0,
            "severity": "none",
            "summary": "No answer was provided to evaluate.",
        }

    matches: List[Dict[str, Any]] = []
    total_count = 0

    for filler, _ in sorted(FILLER_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, text))
        if count:
            matches.append({"word": filler, "count": count})
            total_count += count

    if total_count == 0:
        return {
            "tool": "detect_filler_words",
            "status": "success",
            "filler_words": [],
            "count": 0,
            "severity": "none",
            "summary": "No obvious filler words were detected.",
        }

    if total_count >= 5:
        severity = "high"
    elif total_count >= 2:
        severity = "medium"
    else:
        severity = "low"

    return {
        "tool": "detect_filler_words",
        "status": "success",
        "filler_words": [match["word"] for match in matches],
        "count": total_count,
        "detail": matches,
        "severity": severity,
        "summary": (
            f"Detected {total_count} filler word(s). "
            f"Try pausing for a beat and speaking more directly."
        ),
    }


def check_star_structure(answer: str) -> Dict[str, Any]:
    """Check whether an answer covers the STAR behavioral structure."""
    text = _normalize_text(answer)
    if not text:
        return {
            "tool": "check_star_structure",
            "status": "no_answer",
            "covered": [],
            "missing": ["Situation", "Task", "Action", "Result"],
            "score": 0,
            "summary": "No answer was provided to assess for STAR structure.",
        }

    patterns = {
        "Situation": [
            r"\b(situation|when|during|while|at that time|in my role|back then)\b",
            r"\b(i was|we were|my team|our team)\b",
        ],
        "Task": [
            r"\b(task|challenge|goal|problem|need(ed)? to|responsibility|objective)\b",
            r"\b(had to|needed to|wanted to|required me to)\b",
        ],
        "Action": [
            r"\b(i (built|created|led|improved|managed|launched|collaborated|implemented|resolved|worked|took)\b)",
            r"\b(next|then|after that|first|later|finally)\b",
        ],
        "Result": [
            r"\b(result|outcome|impact|improved|increased|reduced|saved|delivered|achieved|grew|cut|boosted)\b",
            r"\b\d+%|\d+\s*(x|times|people|teams|weeks|days|months|hours|percent)\b",
        ],
    }

    covered: List[str] = []
    missing: List[str] = []

    for section, regexes in patterns.items():
        found = any(re.search(pattern, text) for pattern in regexes)
        if found:
            covered.append(section)
        else:
            missing.append(section)

    score = round((len(covered) / 4) * 100)

    if not missing:
        summary = "Strong STAR structure: the answer covers Situation, Task, Action, and Result."
    elif len(covered) >= 3:
        summary = f"Good structure overall. Add a little more detail on {', '.join(missing)}."
    elif len(covered) >= 2:
        summary = "The response has a decent story arc, but it would be stronger with clearer structure."
    else:
        summary = "The answer would be much stronger with a clearer STAR structure."

    return {
        "tool": "check_star_structure",
        "status": "success",
        "covered": covered,
        "missing": missing,
        "score": score,
        "summary": summary,
    }


def score_relevance(answer: str, expected_keywords: Optional[List[str]] = None, question: Optional[str] = None) -> Dict[str, Any]:
    """Score how much of the expected keyword set is reflected in the answer."""
    answer_text = _normalize_text(answer)
    if not answer_text:
        return {
            "tool": "score_relevance",
            "status": "no_answer",
            "score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "expected_keywords": expected_keywords or [],
            "summary": "No answer was provided to score for relevance.",
        }

    if expected_keywords is None:
        if question is not None:
            expected_keywords = _extract_keywords_from_question(question)
        else:
            expected_keywords = []

    normalized_keywords: List[str] = []
    for keyword in expected_keywords or []:
        cleaned = re.sub(r"[^a-z0-9\s-]", "", str(keyword).lower()).strip()
        if cleaned:
            normalized_keywords.append(cleaned)

    if not normalized_keywords:
        normalized_keywords = _extract_keywords_from_question(question or answer_text)

    if not normalized_keywords:
        return {
            "tool": "score_relevance",
            "status": "success",
            "score": 100,
            "matched_keywords": [],
            "missing_keywords": [],
            "expected_keywords": [],
            "summary": "No explicit keyword set was provided, so no relevance check was possible.",
        }

    answer_tokens = _strip_stopwords(re.findall(r"[a-z0-9-]+", answer_text))
    answer_set = set(answer_tokens)

    matched: List[str] = []
    missing: List[str] = []

    for keyword in normalized_keywords:
        keyword_tokens = _strip_stopwords(keyword.split())
        if not keyword_tokens:
            continue

        # Match phrase or concept-level overlap.
        keyword_phrase = " ".join(keyword_tokens)
        if keyword_phrase in answer_text or all(token in answer_set for token in keyword_tokens):
            matched.append(keyword)
        else:
            missing.append(keyword)

    score = round((len(matched) / max(len(normalized_keywords), 1)) * 100)
    if score >= 80:
        summary = "Strong alignment with the main concepts from the question."
    elif score >= 60:
        summary = "Good coverage of the key ideas, with a few missing points to add."
    elif score >= 35:
        summary = "The answer touches on some relevant points, but it still needs more content tied to the question."
    else:
        summary = "The answer is not yet closely aligned with the expected points from the question."

    return {
        "tool": "score_relevance",
        "status": "success",
        "score": score,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "expected_keywords": normalized_keywords,
        "summary": summary,
    }


def _extract_keywords_from_question(question: Optional[str]) -> List[str]:
    if not question:
        return []

    text = re.sub(r"[^a-z0-9\s-]", " ", str(question).lower())
    tokens = re.findall(r"[a-z0-9-]+", text)
    keywords = _strip_stopwords(tokens)

    if not keywords:
        return []

    # Add a few longer concepts if the question contains them.
    combined = []
    for token in keywords:
        if token not in STOP_WORDS:
            combined.append(token)
    return combined


def decide_relevant_tools(question: Optional[str], answer: str) -> List[str]:
    """Choose which evaluation tools fit the question and answer pair."""
    q = _normalize_text(question)
    tools: List[str] = ["detect_filler_words", "score_relevance"]

    if not q:
        return tools

    behavioral_markers = [
        "tell me about a time",
        "describe a time",
        "give me an example",
        "behavioral",
        "situation",
        "challenge",
        "conflict",
        "leadership",
        "example of",
    ]

    if any(marker in q for marker in behavioral_markers):
        tools.insert(1, "check_star_structure")

    # Deduplicate while maintaining a stable order.
    seen = set()
    ordered: List[str] = []
    for tool in tools:
        if tool not in seen:
            seen.add(tool)
            ordered.append(tool)
    return ordered


def call_groq_for_feedback(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Groq if a key is available and return a concise LLM summary."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def evaluate_answer(question: str, answer: str, expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """Rule-based evaluator agent: choose tools, run them, and generate feedback."""
    tools_used = decide_relevant_tools(question, answer)
    results: Dict[str, Any] = {}

    for tool_name in tools_used:
        if tool_name == "detect_filler_words":
            results[tool_name] = detect_filler_words(answer)
        elif tool_name == "check_star_structure":
            results[tool_name] = check_star_structure(answer)
        elif tool_name == "score_relevance":
            results[tool_name] = score_relevance(answer, expected_keywords=expected_keywords, question=question)

    filler = results.get("detect_filler_words", {})
    star = results.get("check_star_structure", {})
    relevance = results.get("score_relevance", {})

    feedback_parts: List[str] = []

    if filler.get("count", 0) >= 3:
        feedback_parts.append("You have a few filler words, so try pausing before you answer to sound more confident.")
    elif filler.get("count", 0):
        feedback_parts.append("Your answer is mostly clear; a couple of filler phrases can be trimmed for polish.")

    if star and star.get("missing"):
        missing = star.get("missing", [])
        if missing:
            missing_label = ", ".join(missing)
            feedback_parts.append(f"A stronger story would include more detail on {missing_label} to make the example easier to follow.")

    if relevance.get("score", 100) < 70:
        feedback_parts.append("You are close. Add more of the key ideas from the question to show stronger alignment.")
    elif relevance.get("score", 100) >= 80:
        feedback_parts.append("Nice job tying your answer back to the key points in the question.")

    if not feedback_parts:
        feedback_parts.append("Strong answer overall—you communicated your point clearly and confidently.")

    summary = " ".join(feedback_parts)

    llm_feedback = None
    if os.getenv("GROQ_API_KEY"):
        llm_feedback = call_groq_for_feedback(
            "You help interview candidates by giving short, encouraging and specific feedback.",
            f"Question: {question}\nAnswer: {answer}\nTool results: {results}",
        )

    return {
        "question": question,
        "answer": answer,
        "tools_used": tools_used,
        "results": results,
        "feedback": summary,
        "llm_feedback": llm_feedback,
    }


if __name__ == "__main__":
    sample_question = "Tell me about a time you solved a difficult problem at work."
    sample_answer = "Um, I was working on a project and we had a big issue. I needed to fix it quickly. I organized the team, looked at the root cause, and built a plan. As a result, we reduced errors by 30 percent and shipped on time."
    print(evaluate_answer(sample_question, sample_answer))
