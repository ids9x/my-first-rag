"""
Batch query script — run multiple questions through the RAG pipeline.

Workflow (two-step or combined):
  1. classify: LLM analyzes each question and writes a *_classified.yaml file
  2. run:      Execute questions from a (classified) YAML and write results to JSON + CSV
  3. auto:     Classify + run in one shot (reads original, runs from classified copy)

The original questions YAML is never modified.

Usage:
    python3 -m scripts.batch_query classify questions.yaml          # -> questions_classified.yaml
    python3 -m scripts.batch_query classify questions.yaml --force
    python3 -m scripts.batch_query run questions_classified.yaml
    python3 -m scripts.batch_query run questions_classified.yaml --mode hybrid -o results/
    python3 -m scripts.batch_query run questions.yaml --stop-on-error
    python3 -m scripts.batch_query auto questions.yaml
    python3 -m scripts.batch_query auto questions.yaml -o results/ --stop-on-error
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROMPT_PRESETS, DEFAULT_PROMPT_PRESET, ROUTER_TEMPERATURE
from core.llm import get_llm
from modules.router import parse_llm_json

console = Console()

# ---------------------------------------------------------------------------
# Batch-specific classification (extends router.py's 3 categories to 5)
# ---------------------------------------------------------------------------

BATCH_CLASSIFICATION_PROMPT = """\
You are a query classifier for a document retrieval system.
Classify the user question into exactly one category:

- factual_lookup: Direct factual questions, specific section/clause lookups, \
definitions, single-document queries. Best answered with simple vector search.
- comparative: Comparing requirements across documents, standards, or sections. \
Questions that mention two or more standards, clauses, or jurisdictions. \
Benefits from hybrid (vector + keyword) search.
- exploratory: Open-ended analysis, multi-step reasoning, implications, \
"explain why" or "what are the consequences" questions. Needs agentic multi-step reasoning.
- synthesis: Questions requiring thorough analysis across many document sections, \
comprehensive summaries, or questions where missing information from any single \
chunk would be harmful. Benefits from map-reduce (every chunk gets individual LLM attention).
- broad_coverage: Very broad questions where maximum retrieval coverage matters, \
questions spanning multiple topic areas, or questions where no single retrieval \
strategy is clearly best. Benefits from parallel multi-strategy retrieval.

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{{"category": "...", "reasoning": "one sentence explaining why"}}

Question: {question}"""

CATEGORY_TO_MODE = {
    "factual_lookup": "basic",
    "comparative": "hybrid",
    "exploratory": "agentic",
    "synthesis": "map_reduce",
    "broad_coverage": "parallel",
}

# Map mode names to QueryService method names
MODE_TO_METHOD = {
    "basic": "query_basic",
    "hybrid": "query_hybrid",
    "agentic": "query_agentic",
    "router": "query_router",
    "map_reduce": "query_map_reduce",
    "parallel": "query_parallel",
}

VALID_MODES = set(MODE_TO_METHOD.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchQuestion:
    question: str
    mode: Optional[str] = None
    classification_reasoning: Optional[str] = None


@dataclass
class BatchConfig:
    questions: list[BatchQuestion]
    preset: Optional[str] = None
    system_prompt: Optional[str] = None
    output_dir: str = "."


@dataclass
class QuestionResult:
    question_number: int
    question: str
    mode_used: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    reasoning_steps: Optional[list[str]] = None
    routed_to: Optional[str] = None
    classification_reasoning: Optional[str] = None
    map_summaries: Optional[list[str]] = None
    chunk_count: Optional[int] = None
    strategy_counts: Optional[dict] = None
    total_unique_chunks: Optional[int] = None


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_batch_yaml(path: Path) -> BatchConfig:
    """Load a batch YAML file into a BatchConfig."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        console.print("[red]Empty YAML file.[/red]")
        sys.exit(1)

    # Handle flat list (just questions, no settings)
    if isinstance(data, list):
        data = {"questions": data}

    settings = data.get("settings", {})
    raw_questions = data.get("questions", [])

    if not raw_questions:
        console.print("[red]No questions found in YAML file.[/red]")
        sys.exit(1)

    questions = []
    for item in raw_questions:
        if isinstance(item, str):
            questions.append(BatchQuestion(question=item))
        elif isinstance(item, dict):
            questions.append(BatchQuestion(
                question=item["question"],
                mode=item.get("mode"),
                classification_reasoning=item.get("classification_reasoning"),
            ))
        else:
            console.print(f"[yellow]Skipping invalid question entry: {item}[/yellow]")

    # Resolve system prompt
    preset = settings.get("preset")
    system_prompt = settings.get("system_prompt")
    if system_prompt is None and preset and preset in PROMPT_PRESETS:
        system_prompt = PROMPT_PRESETS[preset]

    return BatchConfig(
        questions=questions,
        preset=preset,
        system_prompt=system_prompt,
        output_dir=settings.get("output_dir", "."),
    )


def save_batch_yaml(path: Path, config: BatchConfig) -> None:
    """Write the BatchConfig back to YAML, preserving classified modes."""
    data = {}

    # Settings block
    settings = {}
    if config.preset:
        settings["preset"] = config.preset
    if config.output_dir != ".":
        settings["output_dir"] = config.output_dir
    if settings:
        data["settings"] = settings

    # Questions block
    questions_out = []
    for bq in config.questions:
        entry = {"question": bq.question}
        if bq.mode:
            entry["mode"] = bq.mode
        if bq.classification_reasoning:
            entry["classification_reasoning"] = bq.classification_reasoning
        questions_out.append(entry)

    data["questions"] = questions_out

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)

    console.print(f"[green]Updated:[/green] {path}")


# ---------------------------------------------------------------------------
# Classify subcommand
# ---------------------------------------------------------------------------

def classify_single(question: str, llm) -> dict:
    """Classify a single question using the batch-specific 5-category prompt."""
    prompt_text = BATCH_CLASSIFICATION_PROMPT.format(question=question)

    try:
        response = llm.invoke(prompt_text)
        raw_text = response.content
        result = parse_llm_json(raw_text)

        category = result.get("category", "").strip().lower()
        if category not in CATEGORY_TO_MODE:
            category = "comparative"  # safe fallback

        reasoning = result.get("reasoning", "No reasoning provided")
        return {"category": category, "reasoning": reasoning}

    except Exception as e:
        console.print(f"[yellow]Classification failed: {e}, falling back to hybrid[/yellow]")
        return {"category": "comparative", "reasoning": f"Fallback: {e}"}


def classified_path_for(path: Path) -> Path:
    """Return the classified output path: e.g. questions.yaml -> questions_classified.yaml."""
    return path.with_stem(path.stem + "_classified")


def cmd_classify(args):
    """Classify all questions in the YAML and write a separate classified file."""
    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    config = load_batch_yaml(path)

    # Count how many need classification
    to_classify = [
        bq for bq in config.questions
        if bq.mode is None or args.force
    ]

    if not to_classify:
        console.print("[green]All questions already classified. Use --force to re-classify.[/green]")
        return

    console.print(f"\n[bold]Classifying {len(to_classify)} of {len(config.questions)} questions[/bold]\n")

    llm = get_llm(temperature=ROUTER_TEMPERATURE)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying...", total=len(to_classify))

        for bq in config.questions:
            if bq.mode is not None and not args.force:
                continue

            truncated = bq.question[:60] + "..." if len(bq.question) > 60 else bq.question
            progress.update(task, description=f"  {truncated}")

            result = classify_single(bq.question, llm)
            bq.mode = CATEGORY_TO_MODE[result["category"]]
            bq.classification_reasoning = result["reasoning"]

            progress.advance(task)

    # Save classified output to a separate file (never overwrite the original)
    out_path = classified_path_for(path)
    save_batch_yaml(out_path, config)

    # Print summary table
    mode_counts = {}
    for bq in config.questions:
        mode_counts[bq.mode] = mode_counts.get(bq.mode, 0) + 1

    table = Table(title="Classification Summary")
    table.add_column("Mode", style="bold")
    table.add_column("Count")
    for mode, count in sorted(mode_counts.items()):
        table.add_row(mode, str(count))
    table.add_row("Total", str(len(config.questions)), style="bold")
    console.print(table)


# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------

def execute_question(service, bq: BatchQuestion, question_number: int,
                     default_mode: str, system_prompt: Optional[str]) -> QuestionResult:
    """Execute a single question. Never raises — captures errors in the result."""
    mode = bq.mode or default_mode
    if mode not in MODE_TO_METHOD:
        return QuestionResult(
            question_number=question_number, question=bq.question,
            mode_used=mode, answer="", error=f"Unknown mode: {mode}",
        )

    method_name = MODE_TO_METHOD[mode]
    start = time.time()

    try:
        method = getattr(service, method_name)
        result = method(bq.question, system_prompt=system_prompt)
        duration = time.time() - start

        return QuestionResult(
            question_number=question_number,
            question=bq.question,
            mode_used=result.get("mode", mode),
            answer=result["answer"],
            sources=result.get("sources", []),
            duration_seconds=round(duration, 2),
            reasoning_steps=result.get("reasoning_steps"),
            routed_to=result.get("routed_to"),
            classification_reasoning=result.get("classification_reasoning"),
            map_summaries=result.get("map_summaries"),
            chunk_count=result.get("chunk_count"),
            strategy_counts=result.get("strategy_counts"),
            total_unique_chunks=result.get("total_unique_chunks"),
        )
    except Exception as e:
        duration = time.time() - start
        return QuestionResult(
            question_number=question_number, question=bq.question,
            mode_used=mode, answer="", error=str(e),
            duration_seconds=round(duration, 2),
        )


def write_json(results: list[QuestionResult], metadata: dict, path: Path) -> None:
    """Write full results to JSON."""
    output = {
        "metadata": metadata,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"[green]JSON:[/green] {path}")


def write_csv(results: list[QuestionResult], path: Path) -> None:
    """Write flattened results to CSV."""
    fieldnames = [
        "question_number", "question", "mode_used", "answer",
        "sources", "error", "duration_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # JSON-encode complex fields
            row["sources"] = json.dumps(row["sources"], ensure_ascii=False) if row["sources"] else ""
            writer.writerow(row)
    console.print(f"[green]CSV:[/green]  {path}")


def cmd_run(args):
    """Execute all questions and write results."""
    from core.query_service import QueryService

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    config = load_batch_yaml(path)

    # CLI overrides
    default_mode = args.mode or "basic"
    system_prompt = config.system_prompt
    stop_on_error = args.stop_on_error
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for unclassified questions
    unclassified = sum(1 for bq in config.questions if bq.mode is None)
    if unclassified and not args.mode:
        console.print(
            f"[yellow]{unclassified} question(s) have no mode assigned. "
            f"They will use default mode: {default_mode}[/yellow]"
        )
        console.print("[dim]Tip: run 'classify' first, or use --mode to set a global mode.[/dim]\n")

    # Print run header
    console.print(f"\n[bold]Batch Run[/bold]")
    console.print(f"  Questions:    {len(config.questions)}")
    console.print(f"  Default mode: {default_mode}" + (" (global override)" if args.mode else ""))
    if config.preset:
        console.print(f"  Preset:       {config.preset}")
    console.print(f"  Output:       {output_dir}/")
    console.print()

    # Initialize QueryService once
    service = QueryService()
    results: list[QuestionResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running...", total=len(config.questions))

        for i, bq in enumerate(config.questions, 1):
            # If --mode is set globally, override per-question modes
            effective_bq = bq
            if args.mode:
                effective_bq = BatchQuestion(question=bq.question, mode=args.mode)

            truncated = bq.question[:60] + "..." if len(bq.question) > 60 else bq.question
            mode_label = effective_bq.mode or default_mode
            progress.update(task, description=f"Q{i} [{mode_label}]: {truncated}")

            result = execute_question(service, effective_bq, i, default_mode, system_prompt)
            results.append(result)

            progress.advance(task)

            if result.error:
                console.print(f"  [red]Q{i} error: {result.error}[/red]")
                if stop_on_error:
                    console.print("[red]Stopping on error.[/red]")
                    break

    # Write output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    succeeded = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    total_time = sum(r.duration_seconds for r in results)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "total_duration_seconds": round(total_time, 2),
        "default_mode": default_mode,
        "preset": config.preset,
    }

    json_path = output_dir / f"batch_results_{timestamp}.json"
    csv_path = output_dir / f"batch_results_{timestamp}.csv"
    write_json(results, metadata, json_path)
    write_csv(results, csv_path)

    # Print summary
    console.print()
    table = Table(title="Batch Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total questions", str(len(results)))
    table.add_row("Succeeded", f"[green]{succeeded}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed else "0")
    table.add_row("Total time", f"{total_time:.1f}s")
    table.add_row("Avg time/question", f"{total_time / len(results):.1f}s" if results else "N/A")
    console.print(table)


# ---------------------------------------------------------------------------
# Auto subcommand (classify + run in one shot)
# ---------------------------------------------------------------------------

def cmd_auto(args):
    """Classify all questions, then immediately run them."""
    # Build a namespace that cmd_classify expects
    classify_args = argparse.Namespace(file=args.file, force=args.force)
    cmd_classify(classify_args)

    console.print("\n[bold]--- Classification complete, starting batch run ---[/bold]\n")

    # Point cmd_run at the classified file (original is never modified)
    classified_file = str(classified_path_for(Path(args.file)))
    run_args = argparse.Namespace(
        file=classified_file,
        mode=args.mode,
        output_dir=args.output_dir,
        stop_on_error=args.stop_on_error,
    )
    cmd_run(run_args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch query the RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # classify subcommand
    p_classify = subparsers.add_parser(
        "classify",
        help="LLM-classify each question and assign the best query mode",
    )
    p_classify.add_argument("file", help="YAML file with questions")
    p_classify.add_argument(
        "--force", action="store_true",
        help="Re-classify questions that already have a mode",
    )

    # run subcommand
    p_run = subparsers.add_parser(
        "run",
        help="Execute all questions and write results to JSON + CSV",
    )
    p_run.add_argument("file", help="YAML file with questions")
    p_run.add_argument(
        "--mode", choices=sorted(VALID_MODES),
        help="Override mode for ALL questions",
    )
    p_run.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: from YAML or current dir)",
    )
    p_run.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop on first error (default: continue)",
    )

    # auto subcommand (classify + run combined)
    p_auto = subparsers.add_parser(
        "auto",
        help="Classify questions then immediately run them (both steps)",
    )
    p_auto.add_argument("file", help="YAML file with questions")
    p_auto.add_argument(
        "--force", action="store_true",
        help="Re-classify questions that already have a mode",
    )
    p_auto.add_argument(
        "--mode", choices=sorted(VALID_MODES), default=None,
        help="Override mode for ALL questions during run (skips classification for overridden questions)",
    )
    p_auto.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: from YAML or current dir)",
    )
    p_auto.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop on first error (default: continue)",
    )

    args = parser.parse_args()

    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "auto":
        cmd_auto(args)


if __name__ == "__main__":
    main()
