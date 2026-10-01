import json
import os
import time
import textwrap
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()

console = Console()

MODEL = "gpt-4o-mini"
ITERATIONS = 15

# the schema the downstream pipeline expects - all three strategies are graded against this
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["Positive", "Negative", "Mixed"]},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentiment", "entities"],
    "additionalProperties": False,
}


def get_openai_client():
    """Initializes and returns the OpenAI API client using the environment variable.

    Returns:
        An authenticated OpenAI client instance.

    Side effects:
        Reads OPENAI_API_KEY from the environment.

    Edge cases / gotchas:
        Raises ValueError if OPENAI_API_KEY is not found.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set."
        )
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def call_model(client, prompt, response_format=None, model_name=MODEL, max_retries=4):
    """Generates a text response from the model, retrying on transient errors.

    Args:
        client: The authenticated OpenAI client instance.
        prompt: The string text to send to the model.
        response_format: Optional dict passed to the API to enforce output structure.
        model_name: The model identifier to use.
        max_retries: Number of attempts before giving up.

    Returns:
        The stripped string response, or a string starting with 'Error:' if all
        attempts failed.

    Side effects:
        Makes synchronous network requests and sleeps between retries.

    Edge cases / gotchas:
        Does not retry on hard quota/auth failures, only transient rate limits.
    """
    last_err = ""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                # temperature is pinned across all strategies so the prompt/API
                # constraint is the only variable under test
                "temperature": 0.7,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = str(e)
            # a spent quota or a bad key will never succeed - fail fast
            if any(
                hard in last_err
                for hard in ("insufficient_quota", "invalid_api_key", "401", "404")
            ):
                break
            if any(code in last_err for code in ("429", "500", "503")):
                time.sleep(2**attempt * 2)
                continue
            break
    return f"Error: {last_err}"


def parse_downstream(raw):
    """Simulates a strict downstream consumer parsing the model output.

    Args:
        raw: The string returned by the model.

    Returns:
        A tuple (parsed_ok: bool, schema_ok: bool). parsed_ok is True when the
        string is valid JSON. schema_ok is True when it also matches the shape
        the pipeline requires.

    Edge cases / gotchas:
        Deliberately strict - no markdown fence stripping, no repair, no retry.
        This mirrors a real pipeline that calls json.loads() on the response.
    """
    if raw.startswith("Error:"):
        return False, False

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False, False

    if not isinstance(data, dict):
        return True, False

    sentiment = data.get("sentiment")
    entities = data.get("entities")

    schema_ok = (
        isinstance(sentiment, str)
        and sentiment in ("Positive", "Negative", "Mixed")
        and isinstance(entities, list)
        and all(isinstance(e, str) for e in entities)
    )
    return True, schema_ok


def display_exchange(prompt, response):
    """Prints the prompt, the model response and the parser verdict.

    Args:
        prompt: The string text sent to the model.
        response: The string text returned by the model.

    Side effects:
        Writes styled output to stdout via the global rich Console.
    """
    input_elements = []
    messages = [{"role": "user", "content": prompt}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        indent = " " * (len(role) + 2)
        wrapped = textwrap.fill(content, width=82, subsequent_indent=indent)
        input_elements.append(
            Text.assemble((f"{role.upper()}: ", "bold blue"), (wrapped, "blue"))
        )

    console.print(
        Panel(
            Group(*input_elements),
            title="[bold bright_black]Model Input[/bold bright_black]",
            border_style="bright_black",
            padding=(1, 2),
        )
    )
    console.print()

    wrapped_response = textwrap.fill(response, width=82, subsequent_indent="           ")
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

    parsed_ok, schema_ok = parse_downstream(response)
    if parsed_ok and schema_ok:
        verdict = "[bold green]Downstream parser: OK - valid JSON, schema matched[/bold green]"
    elif parsed_ok:
        verdict = "[bold yellow]Downstream parser: PARTIAL - valid JSON but wrong shape[/bold yellow]"
    else:
        verdict = "[bold red]Downstream parser: CRASHED - json.loads() raised[/bold red]"
    console.print(f"  {verdict}")
    console.print()


def run_experiment(client, name, prompt, response_format=None, iterations=ITERATIONS):
    """Executes a prompt multiple times and collects all responses.

    Args:
        client: The authenticated OpenAI client instance.
        name: Display label for the experiment run.
        prompt: The string text to send to the model.
        response_format: Optional dict passed to the API to enforce output structure.
        iterations: Number of total times to run the prompt.

    Returns:
        A list of string responses from the model.

    Side effects:
        Makes multiple synchronous network requests.
        Writes progress indicators and a sample exchange to stdout.
    """
    console.print(f"[bold cyan]-- {name} {'-' * max(4, 50 - len(name))}[/bold cyan]")
    console.print(f"[dim]Running {iterations} iterations...[/dim]")

    responses = []

    console.print("[dim]  Waiting for first response...[/dim]")
    sample_response = call_model(client, prompt, response_format=response_format)
    display_exchange(prompt, sample_response)
    responses.append(sample_response)

    console.print("  [dim]Collecting remaining responses:[/dim]")

    # 1st run was the sample, start loop at run 2 (index 1)
    for i in range(1, iterations):
        run_num = i + 1
        start_time = time.time()
        resp = call_model(client, prompt, response_format=response_format)
        responses.append(resp)
        elapsed = time.time() - start_time

        # dot colour reflects whether the DOWNSTREAM PARSER survived,
        # not merely whether the API call returned 200
        parsed_ok, schema_ok = parse_downstream(resp)
        if parsed_ok and schema_ok:
            dot = "[green]o[/green]"
        elif parsed_ok:
            dot = "[yellow]o[/yellow]"
        else:
            dot = "[red]x[/red]"

        console.print(
            f"  Run #{run_num:02d}: {dot} [dim]{elapsed:4.1f}s[/dim]",
            end="   ",
            highlight=False,
        )
        if run_num % 4 == 0:
            console.print()

    if iterations % 4 != 0:
        console.print()
    console.print()
    return responses


def calculate_metrics(responses):
    """Calculates variance and downstream-reliability metrics for a set of outputs.

    Args:
        responses: A list of string model outputs.

    Returns:
        A dict with 'unique_count', 'consistency_pct', 'parse_pct', 'schema_pct',
        'total' and 'error_count'.

    Side effects:
        Prints a warning to stdout when any API call failed outright.

    Edge cases / gotchas:
        API failures are excluded from the denominator so that infrastructure
        problems are not mistaken for model behaviour.
    """
    errors = [r for r in responses if r.startswith("Error:")]
    valid = [r for r in responses if not r.startswith("Error:")]

    if errors:
        console.print(
            f"[yellow]Warning: {len(errors)} of {len(responses)} API calls failed "
            f"and were excluded.[/yellow]"
        )
        console.print(f"[yellow]First error: {errors[0][:200]}[/yellow]")

    if not valid:
        console.print("[bold red]All API calls failed - no data to report.[/bold red]")
        return {
            "unique_count": 0,
            "consistency_pct": 0.0,
            "parse_pct": 0.0,
            "schema_pct": 0.0,
            "total": 0,
            "error_count": len(errors),
        }

    counts = Counter(valid)
    most_common_count = counts.most_common(1)[0][1]

    results = [parse_downstream(r) for r in valid]
    parse_ok = sum(1 for p, _ in results if p)
    schema_ok = sum(1 for _, s in results if s)

    return {
        "unique_count": len(counts),
        "consistency_pct": (most_common_count / len(valid)) * 100,
        "parse_pct": (parse_ok / len(valid)) * 100,
        "schema_pct": (schema_ok / len(valid)) * 100,
        "total": len(valid),
        "error_count": len(errors),
    }


def log_responses_to_file(filename, prompt, responses):
    """Writes the prompt, responses and per-run parser verdicts to a Markdown file.

    Args:
        filename: The string path to the output Markdown file.
        prompt: The original prompt sent to the model.
        responses: A list of string model outputs.

    Side effects:
        Writes to the local filesystem, overwriting the file if it exists.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Prompt\n\n```text\n{prompt}\n```\n\n")
        f.write("# Responses\n\n")
        for i, resp in enumerate(responses, 1):
            parsed_ok, schema_ok = parse_downstream(resp)
            if parsed_ok and schema_ok:
                status = "OK - valid JSON, schema matched"
            elif parsed_ok:
                status = "PARTIAL - valid JSON, wrong shape"
            else:
                status = "FAILED - json.loads() raised"
            f.write(f"## Run {i} - {status}\n\n```text\n{resp}\n```\n\n")


def pct_cell(value):
    """Formats a percentage with a colour band that reflects pipeline health.

    Args:
        value: A float percentage between 0 and 100.

    Returns:
        A rich-markup string.
    """
    if value >= 99.9:
        return f"[bold green]{value:.0f}%[/bold green]"
    if value >= 50:
        return f"[yellow]{value:.0f}%[/yellow]"
    return f"[bold red]{value:.0f}%[/bold red]"


def main():
    """Runs all three strategies and prints the comparison table.

    Side effects:
        Configures API, makes network calls, writes log files, and writes
        formatted tables to stdout.
    """
    client = get_openai_client()

    console.print(
        Panel.fit(
            "[bold yellow]LLM Variance & Determinism Prototype[/bold yellow]\n"
            "[dim]How output constraints turn an unreliable model into a reliable component.[/dim]\n"
            f"[dim]Model: {MODEL} | temperature 0.7 | {ITERATIONS} runs per strategy[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    base_review = (
        "The new smartphone is amazing, the camera quality is top-notch but the "
        "battery life is a bit disappointing. I love the design though!"
    )

    task = (
        "Extract the sentiment and key entities from this customer review: "
        f"'{base_review}'"
    )

    unconstrained_prompt = task

    constrained_prompt = (
        f"{task}\n"
        "Output only a JSON object with the keys 'sentiment' and 'entities'. "
        "'sentiment' should be a string (Positive, Negative, or Mixed). "
        "'entities' should be a list of strings."
    )

    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "review_extraction",
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        },
    }

    res_a = run_experiment(
        client, "Strategy A: Unconstrained Prompt", unconstrained_prompt
    )
    log_responses_to_file("output-a-unconstrained.md", unconstrained_prompt, res_a)
    metrics_a = calculate_metrics(res_a)

    res_b = run_experiment(
        client, "Strategy B: Prompt-Constrained (asks for JSON)", constrained_prompt
    )
    log_responses_to_file("output-b-prompt-json.md", constrained_prompt, res_b)
    metrics_b = calculate_metrics(res_b)

    res_c = run_experiment(
        client,
        "Strategy C: Schema-Enforced (API guarantees JSON)",
        constrained_prompt,
        response_format=schema_format,
    )
    log_responses_to_file("output-c-schema.md", constrained_prompt, res_c)
    metrics_c = calculate_metrics(res_c)

    console.print(Rule("[bold yellow]Overall Summary[/bold yellow]", style="yellow"))
    console.print()

    table = Table(title="Variance & Pipeline Reliability", show_lines=True)
    table.add_column("Metric", style="bold", min_width=26)
    table.add_column("A: Unconstrained", justify="center", style="cyan")
    table.add_column("B: Prompt-Constrained", justify="center", style="magenta")
    table.add_column("C: Schema-Enforced", justify="center", style="green")

    table.add_row(
        "Unique Responses",
        str(metrics_a["unique_count"]),
        str(metrics_b["unique_count"]),
        str(metrics_c["unique_count"]),
    )
    table.add_row(
        "Consistency Score",
        f"{metrics_a['consistency_pct']:.0f}%",
        f"{metrics_b['consistency_pct']:.0f}%",
        f"{metrics_c['consistency_pct']:.0f}%",
    )
    table.add_row(
        "Parses as JSON",
        pct_cell(metrics_a["parse_pct"]),
        pct_cell(metrics_b["parse_pct"]),
        pct_cell(metrics_c["parse_pct"]),
    )
    table.add_row(
        "Matches Required Schema",
        pct_cell(metrics_a["schema_pct"]),
        pct_cell(metrics_b["schema_pct"]),
        pct_cell(metrics_c["schema_pct"]),
    )

    console.print(table)
    console.print()

    console.print(
        Panel(
            Text.assemble(
                ("What this shows\n\n", "bold"),
                (
                    f"A - {metrics_a['schema_pct']:.0f}% usable. The model answers well, but in prose. "
                    "A downstream json.loads() crashes on every run. The model is not wrong; "
                    "it is unusable as a software component.\n\n",
                    "",
                ),
                (
                    f"B - {metrics_b['schema_pct']:.0f}% usable. Asking for JSON in the prompt fixes most of it. "
                    "Anything short of 100% is a silent failure rate you must write retry "
                    "and repair logic around.\n\n",
                    "",
                ),
                (
                    f"C - {metrics_c['schema_pct']:.0f}% usable. The API enforces the schema at decode time. "
                    "Not persuasion, a structural guarantee. Same model, same temperature.\n\n",
                    "",
                ),
                (
                    "Temperature was 0.7 throughout. Determinism did not come from turning down "
                    "randomness; it came from narrowing what the model was allowed to emit.",
                    "dim",
                ),
            ),
            title="[bold yellow]Takeaway[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()


if __name__ == "__main__":
    main()