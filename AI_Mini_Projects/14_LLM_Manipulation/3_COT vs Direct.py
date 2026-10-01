"""
Prototype: Chain-of-Thought vs. Direct Answer
==============================================
Demonstrates that asking a model to "think step by step" is a reliability
lever - not just a stylistic choice.

For each multi-step reasoning problem we run two prompt strategies N times:
  - Direct  - "Answer in one short sentence. Do not show working."
  - CoT     - "Think step by step, then state your final answer."

We score every response, aggregate accuracy across all runs, and display
a side-by-side comparison so the *distribution* - not just a single lucky
or unlucky example - is visible.

Concept: CoT forces the model to commit to intermediate steps, which
catches compounding errors before they reach the final answer.

IMPORTANT: use a NON-reasoning model (e.g. gpt-4o-mini). Reasoning models
think internally even when told not to show working, which erases the gap
this prototype exists to measure.
"""

import os
import time
import textwrap

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# find_dotenv() walks up from this file, so a .env in a parent folder is found
load_dotenv(find_dotenv())

console = Console()

MODEL_NAME = "gpt-4o-mini"
RUNS_PER_PROBLEM = 8
TEMPERATURE = 0.7

# how many trailing lines of a response count as the model's "final answer"
TAIL_LINES = 1

# set True to print every prompt/response pair in Rich panels (very verbose:
# one pair per run). False keeps the console output compact, as in output-1.md.
SHOW_EXCHANGES = False

_client = None


def get_client() -> OpenAI:
    """Returns a module-level OpenAI client, creating it on first use.

    Returns:
        An authenticated OpenAI client instance.

    Side effects:
        Reads OPENAI_API_KEY from the environment on first call.

    Edge cases / gotchas:
        Reuses one client across all runs; the original built a fresh client
        per request, which added needless connection overhead.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# -- Problems ------------------------------------------------------------------
# Each problem has a clear answer that is easy to verify automatically.
# The "wrong_answer" field documents the intuitive but incorrect response -
# exactly what a model under direct prompting often produces.

PROBLEMS = [
    {
        "title": "Spatial Navigation (Grid)",
        "question": (
            "Start at origin (0,0) facing North (+Y direction). "
            "Move forward 3 units. Turn right 90 degrees. Move forward 2 units. "
            "Turn right 90 degrees. Move forward 5 units. "
            "Turn left 90 degrees. Move backward 2 units. "
            "What are your exact final (X,Y) coordinates? Format as (X, Y)."
        ),
        "correct_answers": ["(0, -2)", "(0,-2)"],
        "wrong_answer": "(4, -2) or (2, 2) (loses track of current heading before moving)",
        "explanation": "Start(0,0) N -> Fwd 3 to (0,3). Right to E -> Fwd 2 to (2,3). Right to S -> Fwd 5 to (2,-2). Left to E -> Back 2 (moving West) to (0,-2).",
    },
    {
        "title": "Relational Logic (Family Tree)",
        "question": (
            "Alice is the sister of Bob. Bob is the father of Charlie. "
            "Charlie is the brother of Diana. Diana is the mother of Eve. "
            "What is the exact biological relationship of Alice to Eve?"
        ),
        "correct_answers": ["great-aunt", "great aunt", "grand-aunt", "grand aunt"],
        "wrong_answer": "Aunt or Grandmother (skips a generational layer)",
        "explanation": "Bob is Diana's father. Alice is Bob's sister, making her Diana's aunt. Diana is Eve's mother, making Alice Eve's great-aunt.",
    },
    # {
    #     "title": "Temporal Scheduling",
    #     "question": (
    #         "Five speakers (A, B, C, D, E) present one after another. "
    #         "E must speak exactly third. B must speak immediately after D. "
    #         "D cannot be the first speaker. C must speak at some point before A. "
    #         "What is the exact sequence of the 5 speakers from first to last? Format as a comma-separated list."
    #     ),
    #     "correct_answers": ["c, a, e, d, b", "c,a,e,d,b"],
    #     "wrong_answer": "C, D, B, E, A (violates E being third or D being first)",
    #     "explanation": "E is 3rd (_ _ E _ _). DB is a block, can't be 1-2 (D not 1st), so must be 4-5 (_ _ E D B). C before A fills 1 and 2 (C A E D B).",
    # },
    # {
    #     "title": "Inventory State Tracking",
    #     "question": (
    #         "An empty box is given to you. You put in an Apple, a Banana, and a Carrot. "
    #         "You remove the Apple and add a Date. You remove the Carrot and put the Apple back in. "
    #         "You swap the Banana for an Eggplant. Finally, you take out the Date. "
    #         "List exactly the items currently in the box."
    #     ),
    #     "correct_answers": [
    #         "apple, eggplant",
    #         "eggplant, apple",
    #         "apple and eggplant",
    #         "eggplant and apple",
    #     ],
    #     "wrong_answer": "Apple, Banana, Date (fails to apply all state changes sequentially)",
    #     "explanation": "Add A,B,C -> [A,B,C]. Rem A, Add D -> [B,C,D]. Rem C, Add A -> [A,B,D]. Swap B for E -> [A,D,E]. Rem D -> [A,E].",
    # },
    # {
    #     "title": "Logic Puzzle (Truth-Tellers)",
    #     "question": (
    #         "There are three boxes: X, Y, and Z. Exactly one contains a diamond. "
    #         "Box X says: 'The diamond is in Box Y.' "
    #         "Box Y says: 'The diamond is not in Box Y.' "
    #         "Box Z says: 'The diamond is not in Box X.' "
    #         "Exactly one box's statement is true. Which box contains the diamond? "
    #         "Answer with the exact phrase 'Box X', 'Box Y', or 'Box Z'."
    #     ),
    #     "correct_answers": ["box x"],
    #     "wrong_answer": "Box Y or Z (fails to evaluate the statements against all possible states)",
    #     "explanation": "If the diamond is in X: X is false, Y is true, Z is false. This yields exactly one true statement, satisfying the condition.",
    # },
]

# -- Prompt strategies ---------------------------------------------------------


def direct_prompt(question: str) -> str:
    """Returns a prompt requesting a direct answer without reasoning.

    Forces the model to skip intermediate steps, which often leads to errors
    in multi-step logic problems.
    """
    return (
        f"{question}\n\n"
        "Answer in one short sentence only. Do not show any working or reasoning."
    )


def cot_prompt(question: str) -> str:
    """Returns a prompt requesting step-by-step reasoning (Chain-of-Thought).

    Encourages the model to process intermediate steps explicitly,
    improving reliability for complex reasoning tasks.
    """
    return (
        f"{question}\n\n"
        "Think step by step. Show each step of your reasoning clearly, "
        "then state your final answer on the last line."
    )


# -- API helpers ---------------------------------------------------------------


def ask(prompt: str) -> str:
    """Sends a prompt to the model and returns the verbatim response text.

    Args:
        prompt: The string text to send.

    Returns:
        The stripped response text, or a string starting with 'Error:' on failure.

    Side effects:
        Makes a network request. Prints Rich panels when SHOW_EXCHANGES is True.

    Edge cases / gotchas:
        Never raises - a failed call is scored as incorrect so one bad request
        does not abort an 80-run sweep.
    """
    if SHOW_EXCHANGES:
        console.print(
            Panel(
                Group(
                    Text.assemble(("USER: ", "bold blue"), (prompt, "blue")),
                    Rule(style="bright_black"),
                ),
                title="[bold bright_black]Model Input[/bold bright_black]",
                border_style="bright_black",
                padding=(1, 2),
            )
        )

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        response_text = f"Error: {e}"

    if SHOW_EXCHANGES:
        wrapped_response = textwrap.fill(
            response_text, width=82, subsequent_indent="           "
        )
        console.print(
            Panel(
                Text.assemble(
                    ("ASSISTANT: ", "bold green"), (wrapped_response, "italic")
                ),
                title="[bold bright_black]Model Response[/bold bright_black]",
                border_style="bright_black",
                padding=(1, 2),
                highlight=False,
            )
        )
        console.print()

    return response_text


def is_correct(response: str, correct_answers: list[str]) -> bool:
    """Check whether the model's FINAL answer matches an accepted answer.

    Args:
        response: The verbatim model output.
        correct_answers: Accepted answer strings, compared case-insensitively.

    Returns:
        True when the concluding portion of the response contains an accepted
        answer.

    Edge cases / gotchas:
        Scores only the final non-empty line, not the whole response. A CoT
        trace often raises and then discards a candidate answer mid-reasoning;
        matching anywhere in the text would credit CoT for an answer it
        explicitly rejected, inflating the very number this prototype measures.
        Both prompts instruct the model to put its answer at the end, so the
        tail is the fair place to look. Widen TAIL_LINES if a model you test
        habitually buries its conclusion mid-response.
    """
    if response.startswith("Error:"):
        return False

    lines = [ln for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return False

    tail = "\n".join(lines[-TAIL_LINES:]).lower()
    return any(ans.lower() in tail for ans in correct_answers)


# -- Display helpers -----------------------------------------------------------


def accuracy_bar(correct: int, total: int, width: int = 20) -> Text:
    """Render a filled/empty block bar with colour based on accuracy percentage.

    Green for >= 80%, yellow for >= 50%, and red otherwise.
    """
    pct = correct / total
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct >= 0.8 else "yellow" if pct >= 0.5 else "red"
    return Text(f"{bar}  {correct}/{total}  ({int(pct * 100)}%)", style=color)


def run_strategy(prompt_fn, label: str, problem: dict) -> tuple[list[bool], list[float]]:
    """Run one prompt strategy multiple times and capture results and latency.

    Args:
        prompt_fn: Callable turning a question into a prompt string.
        label: Display label for the progress line.
        problem: The problem dict being tested.

    Returns:
        A tuple (results, times) of per-run correctness flags and elapsed seconds.

    Side effects:
        Makes RUNS_PER_PROBLEM network requests and writes progress to stdout.
    """
    results = []
    times = []
    for run in range(1, RUNS_PER_PROBLEM + 1):
        start_t = time.perf_counter()
        response = ask(prompt_fn(problem["question"]))
        elapsed = time.perf_counter() - start_t

        passed = is_correct(response, problem["correct_answers"])
        results.append(passed)
        times.append(elapsed)

        dot = "[green]●[/green]" if passed else "[red]●[/red]"
        console.print(
            f"  {label} #{run:02d}: {dot} [dim]{elapsed:4.1f}s[/dim]",
            end="   ",
            highlight=False,
        )
        if run % 4 == 0:
            console.print()
        time.sleep(0.4)  # gentle on rate limits
    if RUNS_PER_PROBLEM % 4 != 0:
        console.print()
    return results, times


# -- Main ----------------------------------------------------------------------


def run_demo():
    """Main entry point for the CoT vs Direct Answer comparison prototype.

    Iterates through a set of logic problems, running both strategies
    and displaying comparative metrics and a final summary.

    Side effects:
        Makes network calls and writes formatted tables to stdout.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY not set. Add it to a .env file."
        )
        console.print(
            f"[dim]Looked for .env at: {find_dotenv() or 'no .env found in this folder or any parent'}[/dim]"
        )
        raise SystemExit(1)

    console.print(
        Panel.fit(
            "[bold yellow]Chain-of-Thought vs. Direct Answer[/bold yellow]\n"
            "[dim]Comparison of performance and latency across multiple reasoning problems.[/dim]\n"
            f"[dim]{RUNS_PER_PROBLEM} runs x 2 strategies x {len(PROBLEMS)} problems | {MODEL_NAME}[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    summary: list[dict] = []

    for i, problem in enumerate(PROBLEMS, start=1):
        console.print(
            Rule(
                f"[bold]Problem {i} / {len(PROBLEMS)}: {problem['title']}[/bold]",
                style="white",
            )
        )
        console.print(f"[bold]Q:[/bold] {problem['question']}")
        console.print(
            f"[bold]Correct answer:[/bold] [green]{problem['correct_answers'][0]}[/green]"
        )
        console.print(
            f"[bold]Common wrong answer:[/bold] [red]{problem['wrong_answer']}[/red]"
        )
        console.print(f"[dim]{problem['explanation']}[/dim]\n")

        console.print(
            "[bold cyan]-- Direct prompts --------------------------[/bold cyan]"
        )
        direct_results, direct_times = run_strategy(direct_prompt, "Direct", problem)
        console.print()

        console.print(
            "[bold magenta]-- Chain-of-Thought prompts -----------------[/bold magenta]"
        )
        cot_results, cot_times = run_strategy(cot_prompt, "CoT   ", problem)
        console.print()

        direct_acc = sum(direct_results)
        cot_acc = sum(cot_results)
        direct_avg_t = sum(direct_times) / len(direct_times)
        cot_avg_t = sum(cot_times) / len(cot_times)

        table = Table(
            show_header=True, header_style="bold", padding=(0, 2), show_edge=False
        )
        table.add_column("Strategy", width=10)
        table.add_column("Score", width=6, justify="center")
        table.add_column("Avg Time", width=10, justify="right")
        table.add_column("Distribution (filled = correct)")

        table.add_row(
            "[cyan]Direct[/cyan]",
            f"{direct_acc}/{RUNS_PER_PROBLEM}",
            f"{direct_avg_t:.2f}s",
            accuracy_bar(direct_acc, RUNS_PER_PROBLEM),
        )
        table.add_row(
            "[magenta]CoT[/magenta]",
            f"{cot_acc}/{RUNS_PER_PROBLEM}",
            f"{cot_avg_t:.2f}s",
            accuracy_bar(cot_acc, RUNS_PER_PROBLEM),
        )

        console.print(table)
        console.print()

        summary.append(
            {
                "title": problem["title"],
                "direct": direct_acc,
                "cot": cot_acc,
                "direct_t": direct_avg_t,
                "cot_t": cot_avg_t,
            }
        )

    # -- Overall summary -------------------------------------------------------
    console.print(Rule("[bold yellow]Overall Summary[/bold yellow]", style="yellow"))

    total_runs = RUNS_PER_PROBLEM * len(PROBLEMS)
    summary_table = Table(
        show_header=True, header_style="bold", padding=(0, 2), show_edge=False
    )
    summary_table.add_column("Problem", min_width=24)
    summary_table.add_column("Direct", justify="center")
    summary_table.add_column("CoT", justify="center")
    summary_table.add_column("Avg T (D/C)", justify="center")
    summary_table.add_column("CoT lift", justify="center")
    summary_table.add_column("Distribution")

    total_direct = total_cot = 0
    total_direct_t = total_cot_t = 0.0
    for row in summary:
        lift = row["cot"] - row["direct"]
        if lift > 0:
            lift_str = f"[green]+{lift}[/green]"
        elif lift < 0:
            lift_str = f"[red]{lift}[/red]"
        else:
            lift_str = "[dim]+/-0[/dim]"

        # Mini side-by-side bar for the summary
        d_bar = "█" * row["direct"] + "░" * (RUNS_PER_PROBLEM - row["direct"])
        c_bar = "█" * row["cot"] + "░" * (RUNS_PER_PROBLEM - row["cot"])
        dist = Text()
        dist.append(f"D:{d_bar}", style="cyan")
        dist.append("  ")
        dist.append(f"C:{c_bar}", style="magenta")

        summary_table.add_row(
            row["title"],
            f"{row['direct']}/{RUNS_PER_PROBLEM}",
            f"{row['cot']}/{RUNS_PER_PROBLEM}",
            f"{row['direct_t']:.1f}s / {row['cot_t']:.1f}s",
            lift_str,
            dist,
        )
        total_direct += row["direct"]
        total_cot += row["cot"]
        total_direct_t += row["direct_t"]
        total_cot_t += row["cot_t"]

    summary_table.add_section()
    total_lift = total_cot - total_direct
    lift_color = "green" if total_lift > 0 else "red" if total_lift < 0 else "dim"
    avg_direct_t = total_direct_t / len(PROBLEMS)
    avg_cot_t = total_cot_t / len(PROBLEMS)

    summary_table.add_row(
        "[bold]TOTAL / AVG[/bold]",
        f"[bold]{total_direct}/{total_runs}[/bold]",
        f"[bold]{total_cot}/{total_runs}[/bold]",
        f"[bold]{avg_direct_t:.1f}s / {avg_cot_t:.1f}s[/bold]",
        f"[bold {lift_color}]{'+' if total_lift > 0 else ''}{total_lift}[/bold {lift_color}]",
        "",
    )

    console.print(summary_table)
    console.print()

    accuracy_gain = (total_cot - total_direct) / total_runs * 100
    latency_cost = (avg_cot_t / avg_direct_t - 1) * 100 if avg_direct_t else 0
    console.print(
        f"[bold]Trade-off:[/bold] CoT bought "
        f"[green]{accuracy_gain:+.1f}pp accuracy[/green] for "
        f"[yellow]{latency_cost:+.0f}% latency[/yellow]."
    )
    console.print()


if __name__ == "__main__":
    run_demo()