"""
Prototype: Tool Schema Calibration
==================================
Measures which part of a tool schema an LLM actually routes on.

Three weather tools are offered under a 2x2 of conditions:

              loose descriptions   tight descriptions
  descriptive names      A                  B
  opaque names           C                  D

The earlier version of this prototype varied only the DESCRIPTION, while the
tool names stayed get_weather_current / _forecast / _history. Those names
already encode the exact distinction the descriptions were meant to teach, so
a capable model routes on the name and scores ~100% in both conditions - the
"loose" arm was never actually degraded. Crossing names with descriptions
separates the two signals and shows which one is carrying the routing.

The practical lesson: descriptions look optional right up until someone
renames a tool, or you mount an MCP server whose tools are called query,
search and execute. Condition C is what that feels like.

Note: tool_choice is forced, so the model MUST pick one of the three. That
isolates "which tool" from "should I call a tool at all" - otherwise a model
that answers in plain text scores identically to one that routes wrongly, and
the two failures have completely different fixes.
"""

import os
import textwrap
import time
from collections import Counter

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

MODEL_ID = "gpt-4o-mini"

# temperature 0: the variable under test is the schema text, not sampling noise.
# a deterministic run also makes the comparison reproducible for teaching.
TEMPERATURE = 0

_client = None


def get_client() -> OpenAI:
    """Returns a module-level OpenAI client, creating it on first use.

    Returns:
        An authenticated OpenAI client instance.

    Side effects:
        Reads OPENAI_API_KEY from the environment on first call.

    Edge cases / gotchas:
        Exits with a readable message when the key is missing, rather than a
        bare KeyError at import time.
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


def tool(name, description):
    """Builds one OpenAI tool definition.

    Args:
        name: The function name the model will call.
        description: The prose the model routes on - the variable under test.

    Returns:
        A dict in OpenAI's tools format.

    Edge cases / gotchas:
        Every tool here takes the SAME parameter schema on purpose. Only the
        description differs between the Loose and Tight sets, so any accuracy
        gap is attributable to that text alone.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name."}
                },
                "required": ["location"],
            },
        },
    }


# -- The 2x2 design ------------------------------------------------------------
# The original compared Loose vs Tight DESCRIPTIONS while leaving the tool NAMES
# as get_weather_current / _forecast / _history. Those names already encode the
# exact distinction the descriptions were meant to teach, so the model routes on
# the name and never needs the description - "Loose" was not a degraded
# condition at all. Crossing names with descriptions separates the two signals.

CANONICAL = ("current", "forecast", "history")

NAME_SETS = {
    "descriptive": {
        "current": "get_weather_current",
        "forecast": "get_weather_forecast",
        "history": "get_weather_history",
    },
    # plausible but uninformative - the shape real MCP servers often ship with
    "opaque": {
        "current": "weather_service_a",
        "forecast": "weather_service_b",
        "history": "weather_service_c",
    },
}

DESCRIPTION_SETS = {
    "loose": {
        "current": "Get weather information for a location.",
        "forecast": "Get weather data for a city.",
        "history": "Fetch weather records for a specific area.",
    },
    "tight": {
        "current": (
            "Get CURRENT, REAL-TIME weather conditions for a specific location. "
            "Use only for queries about 'now', 'today', or current status."
        ),
        "forecast": (
            "Get FUTURE weather predictions and forecasts. Use only for queries "
            "about 'tomorrow', 'next week', 'upcoming', or future dates."
        ),
        "history": (
            "Fetch HISTORICAL weather records from the past. Use only for queries "
            "about yesterday, last year, or specific past dates."
        ),
    },
}


def build_condition(name_style, desc_style):
    """Builds the tool list and a name->canonical map for one condition.

    Args:
        name_style: Key into NAME_SETS ('descriptive' or 'opaque').
        desc_style: Key into DESCRIPTION_SETS ('loose' or 'tight').

    Returns:
        A tuple (tools, lookup) where lookup maps the emitted function name
        back to its canonical slot so scoring works across naming schemes.
    """
    names = NAME_SETS[name_style]
    descs = DESCRIPTION_SETS[desc_style]
    tools = [tool(names[c], descs[c]) for c in CANONICAL]
    lookup = {names[c]: c for c in CANONICAL}
    return tools, lookup


QUERIES = [
    # Current (17)
    ("How is the weather in Tokyo right now?", "current"),
    ("Is it raining in London at the moment?", "current"),
    ("Current temperature in Paris?", "current"),
    ("Show me today's weather for NYC.", "current"),
    ("What's the humidity like in Berlin today?", "current"),
    # ("Tell me the weather for Sydney now.", "current"),
    # ("Status of weather in Mumbai.", "current"),
    # ("Is it sunny in Madrid today?", "current"),
    # ("Check weather for Rome.", "current"),
    # ("What's the wind speed in Chicago right now?", "current"),
    # ("Weather report for Seattle currently.", "current"),
    # ("Give me the current weather in Toronto.", "current"),
    # ("Current conditions in Moscow.", "current"),
    # ("Is there snow in Oslo today?", "current"),
    # ("Weather update for Cape Town.", "current"),
    # ("Current sky condition in Seoul.", "current"),
    # ("How hot is it in Dubai now?", "current"),
    # Forecast (17)
    ("What will the weather be like tomorrow in London?", "forecast"),
    ("Will it rain next Tuesday in Paris?", "forecast"),
    ("Forecast for Berlin for the coming weekend.", "forecast"),
    ("Is it going to snow tomorrow in NYC?", "forecast"),
    ("Weather prediction for Tokyo next week.", "forecast"),
    # ("Give me the 5-day forecast for Sydney.", "forecast"),
    # ("How's the weather looking for Madrid on Friday?", "forecast"),
    # ("Will the sun be out tomorrow in Rome?", "forecast"),
    # ("Upcoming weather for Chicago.", "forecast"),
    # ("Forecast for Seattle next Monday.", "forecast"),
    # ("Prediction for Toronto weather tomorrow.", "forecast"),
    # ("Future weather in Moscow for the next 3 days.", "forecast"),
    # ("Expected weather in Oslo tomorrow.", "forecast"),
    # ("Is it going to be windy in Cape Town this weekend?", "forecast"),
    # ("Weather outlook for Seoul next month.", "forecast"),
    # ("Will it be hot in Dubai tomorrow?", "forecast"),
    # ("Forecast for Beijing tomorrow.", "forecast"),
    # History (16)
    ("What was the weather like in London yesterday?", "history"),
    ("Did it rain in Paris last Christmas?", "history"),
    ("Temperature in Berlin on July 4th, 2023.", "history"),
    ("How much snow fell in NYC last winter?", "history"),
    ("Weather records for Tokyo in 1990.", "history"),
    # ("Was it sunny in Sydney on my birthday last year?", "history"),
    # ("History of weather in Madrid for August 2022.", "history"),
    # ("Past weather in Rome for June last year.", "history"),
    # ("What was the weather in Chicago on January 1st?", "history"),
    # ("Historical data for Seattle weather in 2010.", "history"),
    # ("Last week's weather in Toronto.", "history"),
    # ("How cold was it in Moscow in December 2020?", "history"),
    # ("Weather in Oslo yesterday afternoon.", "history"),
    # ("Cape Town weather history for last summer.", "history"),
    # ("Was it raining in Seoul three days ago?", "history"),
    # ("Past temperature in Dubai on Feb 1st.", "history"),
]

SHORT_NAME = {None: "(no call)", "ERROR": "(error)"}


def accuracy_bar(correct, total, width=20):
    """Generates a Rich Text accuracy bar with color-coded success levels.

    The color transitions from red (<50%) to yellow (>=50%) to green (>=80%).
    Returns a rich.text.Text object ready for console printing.
    """
    pct = correct / total if total else 0
    filled = round(pct * width)
    bar = "X" * filled + "." * (width - filled)
    color = "green" if pct >= 0.8 else "yellow" if pct >= 0.5 else "red"
    return Text(f"{bar}  {correct}/{total}  ({int(pct * 100)}%)", style=color)


def display_llm_exchange(query, response_text):
    """Renders a visual representation of the LLM input and output.

    Uses Rich panels and rules to create a clear separation between the
    user's query and the model's generated response.
    """
    messages = [{"role": "user", "content": query}]
    input_elements = []
    for msg in messages:
        role = msg["role"]
        content = (
            msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        )

        if role == "user":
            label_style, content_style = "bold blue", "blue"
        elif role == "assistant":
            label_style, content_style = "bold green", "green"
        else:
            label_style, content_style = "dim", "dim"

        indent = " " * (len(role) + 2)
        wrapped = textwrap.fill(content, width=82, subsequent_indent=indent)

        input_elements.append(
            Text.assemble((f"{role.upper()}: ", label_style), (wrapped, content_style))
        )
        input_elements.append(Rule(style="bright_black"))

    if input_elements:
        input_elements.pop()

    console.print(
        Panel(
            Group(*input_elements),
            title="[bold bright_black]Model Input[/bold bright_black]",
            border_style="bright_black",
            padding=(1, 2),
        )
    )
    console.print()

    wrapped_response = textwrap.fill(
        response_text, width=82, subsequent_indent="           "
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
    console.print()


def select_tool(query, tools, lookup):
    """Asks the model to route one query and returns the tool it picked.

    Args:
        query: The user's natural-language request.
        tools: The tool definitions to offer.
        lookup: Maps each emitted function name to its canonical slot, so
            opaque and descriptive naming schemes score identically.

    Returns:
        The selected canonical slot, None when the model declined to call a
        tool, or 'ERROR' when the request failed.

    Side effects:
        Makes a synchronous network request.

    Edge cases / gotchas:
        tool_choice='required' forces a selection. Without it, a model that
        replies in prose scores the same as one that routes to the wrong tool,
        and those are different bugs with different fixes.
    """
    try:
        response = get_client().chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            tool_choice="required",
            temperature=TEMPERATURE,
        )
        calls = response.choices[0].message.tool_calls
        if not calls:
            return None
        return lookup.get(calls[0].function.name, calls[0].function.name)
    except Exception as e:
        console.print(f"\n[bold red]API error:[/bold red] {e}")
        return "ERROR"


def run_calibration(label, name_style, desc_style, color, show_sample=False):
    """Evaluates routing accuracy for one (names x descriptions) condition.

    Args:
        label: Display name for this condition.
        name_style: Key into NAME_SETS.
        desc_style: Key into DESCRIPTION_SETS.
        color: Rich colour for the section header.
        show_sample: Print the first exchange as a rendered panel.

    Returns:
        A tuple (correct, total, results) where results is a list of
        (query, expected, selected, passed) tuples.

    Side effects:
        Makes one network request per query and writes progress to stdout.
    """
    tools, lookup = build_condition(name_style, desc_style)

    console.print(
        f"[bold {color}]-- {label} "
        f"--------------------------------------[/bold {color}]"
    )
    console.print(
        f"[dim]  names: {name_style}  |  descriptions: {desc_style}[/dim]"
    )
    console.print()

    correct = 0
    total = len(QUERIES)
    results = []

    for i, (query, expected) in enumerate(QUERIES):
        start_time = time.time()
        selected = select_tool(query, tools, lookup)
        elapsed = time.time() - start_time

        passed = selected == expected
        if passed:
            correct += 1
        results.append((query, expected, selected, passed))

        if i == 0 and show_sample:
            display_llm_exchange(query, f"Selected tool: {selected}")

        dot = "[green]o[/green]" if passed else "[red]o[/red]"
        console.print(
            f"  Run #{i + 1:02d}: {dot} [dim]{elapsed:4.1f}s[/dim]",
            end="   ",
            highlight=False,
        )
        if (i + 1) % 5 == 0:
            console.print()

    console.print()
    console.print(f"  {label}: ", end="")
    console.print(accuracy_bar(correct, total))
    console.print()
    return correct, total, results


def confusion_table(results, title):
    """Builds a per-category breakdown of where routing went wrong.

    Args:
        results: The list of (query, expected, selected, passed) tuples.
        title: Table title.

    Returns:
        A rich Table showing, for each expected tool, what the model picked.

    Edge cases / gotchas:
        A single accuracy number hides the failure MODE. Vague descriptions
        typically collapse everything onto one tool, and that collapse is the
        thing worth showing - it tells you which description to rewrite.
    """
    table = Table(title=title, show_lines=False)
    table.add_column("Expected", style="bold")
    table.add_column("Correct", justify="center")
    table.add_column("Routed to instead", style="dim")

    for expected in CANONICAL:
        rows = [r for r in results if r[1] == expected]
        hits = sum(1 for r in rows if r[3])
        wrong = Counter(SHORT_NAME.get(r[2], r[2]) for r in rows if not r[3])
        detail = (
            ", ".join(f"{name} x{n}" for name, n in wrong.most_common())
            if wrong
            else "-"
        )
        colour = "green" if hits == len(rows) else "red" if hits == 0 else "yellow"
        table.add_row(
            expected,
            f"[{colour}]{hits}/{len(rows)}[/{colour}]",
            detail,
        )
    return table


def main():
    """Runs all four (names x descriptions) conditions and prints the matrix.

    Side effects:
        Makes network calls and writes tables to stdout.
    """
    conditions = [
        ("A: descriptive names + loose descs", "descriptive", "loose", "cyan"),
        ("B: descriptive names + tight descs", "descriptive", "tight", "magenta"),
        ("C: opaque names + loose descs", "opaque", "loose", "red"),
        ("D: opaque names + tight descs", "opaque", "tight", "green"),
    ]

    console.print(
        Panel.fit(
            "[bold yellow]Tool Schema Calibration[/bold yellow]\n"
            "[dim]Which field does the model actually route on - the name or the description?[/dim]\n"
            f"[dim]{len(QUERIES)} queries x {len(conditions)} conditions | {MODEL_ID} | temperature {TEMPERATURE}[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    scores = {}
    all_results = {}
    total = len(QUERIES)

    for idx, (label, name_style, desc_style, color) in enumerate(conditions):
        correct, total, results = run_calibration(
            label, name_style, desc_style, color, show_sample=(idx == 0)
        )
        scores[(name_style, desc_style)] = correct
        all_results[(name_style, desc_style)] = results

    console.print(Rule("[bold yellow]Overall Summary[/bold yellow]", style="yellow"))
    console.print()

    # -- the 2x2 matrix --------------------------------------------------------
    matrix = Table(title="Routing accuracy: names x descriptions", show_lines=True)
    matrix.add_column("Tool names", style="bold", min_width=18)
    matrix.add_column("Loose descriptions", justify="center", min_width=22)
    matrix.add_column("Tight descriptions", justify="center", min_width=22)

    for name_style in ("descriptive", "opaque"):
        row = [f"{name_style}"]
        for desc_style in ("loose", "tight"):
            c = scores[(name_style, desc_style)]
            pct = c / total * 100
            colour = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            row.append(f"[{colour}]{c}/{total}  ({pct:.0f}%)[/{colour}]")
        matrix.add_row(*row)

    console.print(matrix)
    console.print()

    # -- what the matrix says --------------------------------------------------
    desc_lift_descriptive = (
        scores[("descriptive", "tight")] - scores[("descriptive", "loose")]
    )
    desc_lift_opaque = scores[("opaque", "tight")] - scores[("opaque", "loose")]
    name_lift_loose = (
        scores[("descriptive", "loose")] - scores[("opaque", "loose")]
    )

    lines = [
        f"Description lift, when names are self-explanatory: "
        f"[{'green' if desc_lift_descriptive > 0 else 'dim'}]{desc_lift_descriptive:+d}[/]",
        f"Description lift, when names are opaque:           "
        f"[{'green' if desc_lift_opaque > 0 else 'dim'}]{desc_lift_opaque:+d}[/]",
        f"Name lift, when descriptions are vague:            "
        f"[{'green' if name_lift_loose > 0 else 'dim'}]{name_lift_loose:+d}[/]",
    ]

    console.print(
        Panel.fit(
            "\n".join(lines), title="[bold]Signal attribution[/bold]", border_style="green"
        )
    )
    console.print()

    if desc_lift_opaque > desc_lift_descriptive:
        verdict = (
            "[bold green]Descriptions are load-bearing - but only when the names "
            "are not already doing the work.[/bold green]\n"
            "[dim]With self-explanatory names the model routes on the name and the "
            "description barely matters. Rename your tools generically and the "
            "description becomes the only signal left.[/dim]"
        )
    elif desc_lift_descriptive > 0 or desc_lift_opaque > 0:
        verdict = (
            "[bold yellow]Descriptions help in both conditions.[/bold yellow]\n"
            "[dim]Tightening the prose improves routing regardless of naming.[/dim]"
        )
    else:
        verdict = (
            "[bold cyan]No measurable effect from descriptions here.[/bold cyan]\n"
            "[dim]Either the model is saturating this task, or the queries are too "
            "easy to separate. Add more overlapping tools to raise the difficulty.[/dim]"
        )

    console.print(f"VERDICT: {verdict}")
    console.print()

    # the worst condition is where the failure mode is legible
    worst = min(scores, key=scores.get)
    console.print(
        confusion_table(
            all_results[worst],
            f"Routing breakdown - worst condition ({worst[0]} names, {worst[1]} descs)",
        )
    )
    console.print()

    errors = sum(1 for rs in all_results.values() for r in rs if r[2] == "ERROR")
    if errors:
        console.print(
            f"[yellow]Note: {errors} calls failed and were counted as incorrect.[/yellow]"
        )
        console.print()


if __name__ == "__main__":
    main()