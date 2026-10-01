"""
Prototype: Direct vs. Zero-Shot CoT vs. Few-Shot CoT
====================================================
Benchmarks three prompting strategies on a set of questions chosen because
the intuitive answer is wrong. Reports accuracy AND token cost, so the
accuracy gain can be weighed against what it costs to buy.

IMPORTANT: use a NON-reasoning model (e.g. gpt-4o-mini). Reasoning models
think internally even under a "just the final answer" prompt, which erases
the gap this benchmark exists to measure.
"""

import os
import re
import time
import textwrap

import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# find_dotenv() walks up from this file, so a .env in a parent folder is found
load_dotenv(find_dotenv())

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.7

console = Console()

_client = None


def get_client() -> OpenAI:
    """Returns a module-level OpenAI client, creating it on first use.

    Returns:
        An authenticated OpenAI client instance.

    Side effects:
        Reads OPENAI_API_KEY from the environment on first call.

    Edge cases / gotchas:
        Raises SystemExit with a readable message when the key is missing,
        rather than a bare KeyError at import time.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[bold red]Error:[/bold red] OPENAI_API_KEY not set. Add it to a .env file."
            )
            console.print(
                f"[dim]Looked for .env at: "
                f"{find_dotenv() or 'no .env found in this folder or any parent'}[/dim]"
            )
            raise SystemExit(1)
        _client = OpenAI(api_key=api_key)
    return _client


# Benchmark Questions
# Chosen to require actual multi-step computation. Deliberately NOT famous
# puzzles (bat-and-ball, 17 sheep, 5 machines): those appear verbatim in
# training data, so a direct prompt retrieves a memorised answer rather than
# reasoning, and CoT has no headroom left to show a gain.
BENCHMARK_DATA = [
    {
        "question": "A shop sells pens at 23 rupees each. Ravi buys 7 pens, returns 2 of them, then buys 4 more. He pays the final bill with a 500 rupee note. How much change does he get? (Answer with the number only)",
        "answer": ["293"],
    },
    # {
    #     "question": "A jacket is priced at 4000 rupees. A 25% discount is applied, then a further 10% off the reduced price, and finally 18% GST is added to that result. What is the final price? (Answer with the number only)",
    #     "answer": ["3186"],
    # },
    # {
    #     "question": "A factory produces 2000 units in a day. 8% fail quality control. Of the units that pass, 15% are exported and the rest are sold domestically. How many units are sold domestically? (Answer with the number only)",
    #     "answer": ["1564"],
    # },
    # {
    #     "question": "A train departs at 14:35. The journey takes 3 hours 50 minutes, then it waits 25 minutes at a junction, then continues for a further 1 hour 40 minutes. What time does it arrive? (Answer in 24-hour HH:MM format)",
    #     "answer": ["20:30", "8:30 pm", "8:30pm"],
    # },
    # {
    #     "question": "Three printers together print 90 pages in 6 minutes. At the same rate per printer, how many pages would 5 printers print in 10 minutes? (Answer with the number only)",
    #     "answer": ["250"],
    # },
]

# Few-Shot Examples for reasoning-based prompting
FEW_SHOT_EXAMPLES = """
Example 1:
Question: If I have 3 apples and I give 2 to my friend, but then my friend gives me 1 back, how many apples do I have?
Thought:
1. Start with 3 apples.
2. Give 2 away: 3 - 2 = 1 apple remaining.
3. Friend gives 1 back: 1 + 1 = 2 apples.
Answer: 2

Example 2:
Question: How many legs does a spider have?
Thought:
1. Spiders are arachnids.
2. Arachnids typically have 8 legs.
Answer: 8

Example 3:
Question: What is 15 * 4?
Thought:
1. 15 * 2 is 30.
2. 30 * 2 is 60.
Answer: 60
"""


def get_model_response(prompt):
    """Sends a prompt to the model and returns the response text and token counts.

    Args:
        prompt: The string text to send.

    Returns:
        A tuple (text, prompt_tokens, completion_tokens). On failure, returns
        an 'Error: ...' string with zero token counts.

    Side effects:
        Makes a synchronous network request.

    Edge cases / gotchas:
        Never raises, so a single failed call does not abort the benchmark.
    """
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        usage = response.usage
        return (
            response.choices[0].message.content,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
    except Exception as e:
        return f"Error: {e}", 0, 0


def extract_final_answer(response_text):
    """Isolates the model's stated final answer from a full response.

    Args:
        response_text: The verbatim model output.

    Returns:
        The answer segment as a lowercase string.

    Edge cases / gotchas:
        Prefers the text after the LAST 'Answer:' marker, which both CoT
        prompts request. Falls back to the final non-empty line, which is
        what a Direct response is. This is what stops a reasoning trace from
        being searched for the right answer.
    """
    text = response_text.strip()

    matches = list(re.finditer(r"answer\s*:", text, flags=re.IGNORECASE))
    if matches:
        segment = text[matches[-1].end() :]
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        segment = lines[-1] if lines else ""

    # take only the first line of the segment, and drop markdown emphasis
    segment = segment.strip().splitlines()[0] if segment.strip() else ""
    return segment.replace("*", "").replace("`", "").strip().lower()


def first_clause(segment):
    """Trims a stated answer down to the asserted value, dropping justification.

    Args:
        segment: The lowercase answer segment from extract_final_answer().

    Returns:
        The leading clause as a string.

    Edge cases / gotchas:
        Cuts at a comma, semicolon, sentence break or a joining word such as
        'since'. Without this, a response like 'Answer: 8 sheep are left, since
        17 - 9 = 8' would be searched for the expected '9' and score correct.
        The sentence break is '. ' (period plus space) so decimals like 0.05
        survive intact.
    """
    cut_points = [segment.find(sep) for sep in (",", ";", ". ")]
    for word in (" since ", " because ", " which ", " as ", " so ", " therefore "):
        cut_points.append(segment.find(word))

    valid = [p for p in cut_points if p > 0]
    return segment[: min(valid)].strip() if valid else segment.strip()


def evaluate_accuracy(response_text, correct_answer):
    """Checks whether the model's FINAL answer matches an expected answer.

    Args:
        response_text: The verbatim model output.
        correct_answer: The expected answer, as a string or a list of
            acceptable strings.

    Returns:
        True when the extracted final answer matches any accepted value.

    Edge cases / gotchas:
        The original scorer searched the WHOLE response for the answer as a raw
        substring after stripping '.' and ','. With single-digit answers that
        matched almost anything: a numbered reasoning step '1.' satisfied the
        answer '1', and a response echoing '5 machines' satisfied the answer
        '5'. Every CoT response scored correct regardless of its conclusion,
        inflating exactly the number this benchmark measures. This version
        scores only the stated final answer, with word boundaries, and for a
        numeric answer requires the FIRST number stated to be the answer.
    """
    if response_text.startswith("Error:"):
        return False

    accepted = (
        [correct_answer] if isinstance(correct_answer, str) else list(correct_answer)
    )

    answer_segment = extract_final_answer(response_text)
    if not answer_segment:
        return False

    # keep only the asserted value, dropping any trailing justification such as
    # "8 sheep are left, since 17 - 9 = 8" -> "8 sheep are left"
    clause = first_clause(answer_segment)
    cleaned_clause = clause.replace("$", "").replace(",", "")

    for candidate in accepted:
        cleaned_answer = candidate.strip().lower().replace("$", "").replace(",", "")

        if cleaned_clause.rstrip(".") == cleaned_answer:
            return True

        # for a numeric expected answer, the FIRST number stated must be it.
        # otherwise "100 minutes" would match an expected "5" via a later mention.
        if re.fullmatch(r"\d+(\.\d+)?", cleaned_answer):
            numbers = re.findall(r"\d+(?:\.\d+)?", cleaned_clause)
            if numbers and float(numbers[0]) == float(cleaned_answer):
                return True
            continue

        if re.search(rf"(?<!\w){re.escape(cleaned_answer)}(?!\w)", cleaned_clause):
            return True

    return False


def accuracy_bar(correct: int, total: int, width: int = 20) -> Text:
    """Generates a visual progress bar for accuracy display."""
    pct = correct / total
    filled = round(pct * width)
    bar = "X" * filled + "." * (width - filled)
    color = "green" if pct >= 0.8 else "yellow" if pct >= 0.5 else "red"
    return Text(f"{bar}  {correct}/{total}  ({int(pct * 100)}%)", style=color)


def run_benchmark():
    """Executes the benchmark across three prompting strategies and collects metrics.

    Returns:
        A dict keyed by strategy name holding accuracy, token totals and counts.

    Side effects:
        Makes network requests and writes styled output to stdout.
    """
    results = {
        "Direct": {"accuracy": 0, "tokens": 0, "count": 0},
        "Zero-Shot CoT": {"accuracy": 0, "tokens": 0, "count": 0},
        "Few-Shot CoT": {"accuracy": 0, "tokens": 0, "count": 0},
    }

    strategies = [
        (
            "Direct",
            "Answer the following question directly with just the final answer: {question}",
            "cyan",
        ),
        (
            "Zero-Shot CoT",
            "Answer the following question. Think step-by-step and then provide the final answer as 'Answer: <value>': {question}",
            "magenta",
        ),
        (
            "Few-Shot CoT",
            f"Answer the following question. Think step-by-step and then provide the final answer as 'Answer: <value>'.\n\n{FEW_SHOT_EXAMPLES}\n\nQuestion: {{question}}",
            "yellow",
        ),
    ]

    console.print(
        Panel.fit(
            f"[bold yellow]Prompting Strategy Benchmark[/bold yellow]\n"
            f"[dim]Comparing Direct, Zero-Shot CoT, and Few-Shot CoT performance.[/dim]\n"
            f"[dim]Model: {MODEL_NAME} | Questions: {len(BENCHMARK_DATA)}[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    for name, template, color in strategies:
        console.print(
            f"[bold {color}]-- {name} --------------------------------------[/bold {color}]"
        )
        total_correct = 0
        total_tokens = 0

        for i, item in enumerate(BENCHMARK_DATA, 1):
            prompt = template.format(question=item["question"])

            # Model Input Block
            wrapped_input = textwrap.fill(prompt, width=82, subsequent_indent="      ")
            input_content = Text.assemble(
                ("USER: ", "bold blue"), (wrapped_input, "blue")
            )
            console.print(
                Panel(
                    input_content,
                    title="[bold bright_black]Model Input[/bold bright_black]",
                    border_style="bright_black",
                    padding=(1, 2),
                )
            )

            start_time = time.time()
            response, p_tokens, c_tokens = get_model_response(prompt)
            elapsed = time.time() - start_time
            total_tokens += p_tokens + c_tokens

            expected = item["answer"]
            expected_label = expected[0] if isinstance(expected, list) else expected
            is_correct = evaluate_accuracy(response, expected)
            if is_correct:
                total_correct += 1

            # Model Response Block
            wrapped_response = textwrap.fill(
                response.strip(), width=82, subsequent_indent="           "
            )
            response_content = Text.assemble(
                ("ASSISTANT: ", "bold green"), (wrapped_response, "italic")
            )
            console.print(
                Panel(
                    response_content,
                    title="[bold bright_black]Model Response[/bold bright_black]",
                    border_style="bright_black",
                    padding=(1, 2),
                    highlight=False,
                )
            )

            dot = "[green]o[/green]" if is_correct else "[red]x[/red]"
            scored = extract_final_answer(response)
            console.print(
                f"  [dim]Run #{i:02d}:[/dim] {dot} [dim]{elapsed:4.1f}s | "
                f"{p_tokens + c_tokens} tokens | scored on: '{scored}' "
                f"| expected: '{expected_label}'[/dim]",
                highlight=False,
            )
            console.print()

        results[name]["accuracy"] = (total_correct / len(BENCHMARK_DATA)) * 100
        results[name]["total_correct"] = total_correct
        results[name]["tokens"] = total_tokens
        results[name]["count"] = len(BENCHMARK_DATA)
        console.print()

    return results


def plot_results(results):
    """Generates and saves a performance plot comparing strategies.

    Args:
        results: The dict returned by run_benchmark().

    Side effects:
        Writes benchmark_results.png to the current directory.
    """
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    tokens = [results[n]["tokens"] for n in names]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Accuracy (%)", color=color)
    ax1.bar(names, accuracies, color=color, alpha=0.6, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 110)

    # Secondary axis for token count
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Total Tokens Used", color=color)
    ax2.plot(names, tokens, color=color, marker="o", label="Token Cost")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Accuracy vs. Token Cost ({MODEL_NAME})")
    fig.tight_layout()
    plt.savefig("benchmark_results.png")


def display_stats(results):
    """Displays a summary table and final verdict based on benchmark data.

    Args:
        results: The dict returned by run_benchmark().

    Side effects:
        Writes tables and a verdict panel to stdout.
    """
    console.print(Rule("[bold yellow]Overall Summary[/bold yellow]", style="yellow"))
    console.print()

    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Strategy", style="bold", min_width=15)
    table.add_column("Accuracy", justify="center")
    table.add_column("Total Tokens", justify="center", style="dim")
    table.add_column("Avg Tokens/Query", justify="center", style="dim")

    for name, data in results.items():
        avg_tokens = data["tokens"] / data["count"]
        acc_bar = accuracy_bar(data["total_correct"], data["count"])

        table.add_row(
            name,
            acc_bar,
            str(data["tokens"]),
            f"{avg_tokens:.1f}",
        )

    console.print(table)
    console.print()

    # Evaluate best performing strategies
    best_acc = max(results.values(), key=lambda x: x["accuracy"])["accuracy"]
    best_strategies = [n for n, d in results.items() if d["accuracy"] == best_acc]

    # Calculate token efficiency
    efficiency = {
        n: (d["accuracy"] / (d["tokens"] / 1000)) if d["tokens"] > 0 else 0
        for n, d in results.items()
    }
    most_efficient = max(efficiency, key=efficiency.get)

    baseline = results["Direct"]
    verdict_lines = [
        f"Highest Accuracy: [green]{', '.join(best_strategies)} ({best_acc:.1f}%)[/green]",
        f"Most Efficient:   [cyan]{most_efficient}[/cyan] "
        f"[dim]({efficiency[most_efficient]:.1f} accuracy pts per 1k tokens)[/dim]",
    ]

    # what each strategy cost, relative to answering directly
    for name, data in results.items():
        if name == "Direct" or baseline["tokens"] == 0:
            continue
        acc_delta = data["accuracy"] - baseline["accuracy"]
        token_mult = data["tokens"] / baseline["tokens"]
        verdict_lines.append(
            f"[dim]{name}: {acc_delta:+.0f}pp accuracy for {token_mult:.1f}x the tokens[/dim]"
        )

    console.print(
        Panel.fit(
            "\n".join(verdict_lines),
            title="[bold]Verdict[/bold]",
            border_style="green",
        )
    )
    console.print()
    console.print("[dim]Plot saved to benchmark_results.png[/dim]")


if __name__ == "__main__":
    results = run_benchmark()
    display_stats(results)
    plot_results(results)