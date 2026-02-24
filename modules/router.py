"""
Router module — classifies user queries and dispatches to the best pipeline.

Uses a lightweight LLM call to determine whether the question is a factual
lookup, a comparative analysis, or an exploratory/multi-hop question, then
routes to Basic, Hybrid, or Agentic mode respectively.
"""
import json
import re

from rich.console import Console

from core.llm import get_llm
from config.settings import ROUTER_TEMPERATURE, ROUTER_MAX_TOKENS

console = Console()

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------
# The LLM is asked to return JSON with a category and one-sentence reasoning.
# Few-shot examples use nuclear regulatory language so the model understands
# the domain context.

ROUTE_CLASSIFICATION_PROMPT = """\
You are a query classifier for a nuclear regulatory document retrieval system.
Classify the user question into exactly one category:

- factual_lookup: Direct factual questions, specific section/clause lookups, \
definitions, single-document queries.
- comparative: Comparing requirements across documents, standards, or sections. \
Questions that mention two or more standards, clauses, or jurisdictions.
- exploratory: Open-ended analysis, multi-step reasoning, implications, \
"explain why" or "what are the consequences" questions.

Examples:
- "What is the definition of safety class?" → factual_lookup
- "What does Clause 4.2 of NQA-1 require?" → factual_lookup
- "Compare NQA-1 and ISO 19443 quality requirements" → comparative
- "How do Canadian and US nuclear safety standards differ?" → comparative
- "Explain the implications of removing hold points from inspection plans" → exploratory
- "What are the consequences of non-conformance in nuclear procurement?" → exploratory

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{{"category": "...", "reasoning": "one sentence explaining why"}}

Question: {question}"""


# ---------------------------------------------------------------------------
# Mapping from classification category to QueryService method name
# ---------------------------------------------------------------------------
CATEGORY_TO_METHOD = {
    "factual_lookup": "query_basic",
    "comparative": "query_hybrid",
    "exploratory": "query_agentic",
}

# Human-readable labels for the UI
CATEGORY_TO_LABEL = {
    "factual_lookup": "Basic (Vector)",
    "comparative": "Hybrid (Vector+BM25)",
    "exploratory": "Agentic (Multi-step)",
}


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping wrappers if present.

    Handles common LLM output quirks:
      - Qwen3 <think>...</think> reasoning blocks before the JSON
      - Markdown code fences (```json ... ```)
      - Leading/trailing prose around the JSON object

    Args:
        text: Raw LLM output that may contain wrappers around JSON.

    Returns:
        Parsed dict from the JSON content.
    """
    # 1. Remove <think>...</think> blocks (Qwen3 reasoning mode)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned).strip().rstrip('`')

    # 3. Try direct parse first (fastest path)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Fallback: find the first JSON object {...} in the text
    match = re.search(r'\{[^{}]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 5. Fallback: LLM returned malformed JSON with unquoted values.
    #    Extract category and reasoning with regex instead of JSON parsing.
    #    Example: {"category": "factual_lookup", "reasoning": some unquoted text}
    cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', cleaned)
    if cat_match:
        category = cat_match.group(1)
        # Try to grab reasoning (may be quoted or unquoted)
        reason_match = re.search(r'"reasoning"\s*:\s*"?([^"{}]+)"?\s*\}?', cleaned)
        reasoning = reason_match.group(1).strip() if reason_match else "No reasoning"
        return {"category": category, "reasoning": reasoning}

    # 6. Nothing worked — raise so the caller can fall back
    raise json.JSONDecodeError("No JSON object found in LLM output", cleaned, 0)


def classify_query(question: str, llm=None) -> dict:
    """Classify a user question into a retrieval category.

    Sends the question to the LLM with a structured prompt and parses
    the JSON response. Falls back to "comparative" (Hybrid) if parsing fails,
    since Hybrid is the best general-purpose pipeline.

    Args:
        question: The user's natural-language question.
        llm: Optional pre-configured LLM instance. If None, creates one
             with ROUTER_TEMPERATURE for deterministic classification.

    Returns:
        dict with:
            - category: str ("factual_lookup", "comparative", or "exploratory")
            - reasoning: str (one-sentence explanation from the LLM)
    """
    # Use a low-temperature LLM for deterministic classification
    if llm is None:
        llm = get_llm(temperature=ROUTER_TEMPERATURE)

    # Build the classification prompt with the user's question
    prompt_text = ROUTE_CLASSIFICATION_PROMPT.format(question=question)

    try:
        # Call the LLM — invoke() returns an AIMessage object
        response = llm.invoke(prompt_text)
        raw_text = response.content

        # Debug: show what the LLM actually returned (first 300 chars)
        console.print(
            f"[dim]Router: raw LLM output ({len(raw_text)} chars): "
            f"{repr(raw_text[:300])}[/dim]"
        )

        # Parse the JSON response
        result = parse_llm_json(raw_text)

        # Validate that category is one of the expected values
        category = result.get("category", "").strip().lower()
        if category not in CATEGORY_TO_METHOD:
            console.print(
                f"[yellow]Router: unexpected category '{category}', "
                f"falling back to 'comparative' (Hybrid)[/yellow]"
            )
            category = "comparative"

        reasoning = result.get("reasoning", "No reasoning provided")

        return {"category": category, "reasoning": reasoning}

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # JSON parsing failed — fall back to Hybrid (best general-purpose)
        console.print(
            f"[yellow]Router: failed to parse classification JSON ({e}), "
            f"falling back to 'comparative' (Hybrid)[/yellow]"
        )
        console.print(f"[dim]Router: raw text was: {repr(raw_text[:500])}[/dim]")
        return {
            "category": "comparative",
            "reasoning": f"Fallback — could not parse LLM classification: {e}",
        }

    except Exception as e:
        # LLM call itself failed — fall back to Hybrid
        console.print(f"[red]Router: LLM classification error: {e}[/red]")
        return {
            "category": "comparative",
            "reasoning": f"Fallback — LLM call failed: {e}",
        }


def route_and_execute(question: str, query_service, chat_history=None) -> dict:
    """Classify a question and dispatch to the best query pipeline.

    This is the main entry point called by QueryService.query_router().
    It first classifies the question, then calls the matching query method
    on the QueryService instance, and enriches the result with routing metadata.

    Args:
        question: The user's natural-language question.
        query_service: A QueryService instance (has query_basic, query_hybrid,
                       query_agentic methods).
        chat_history: Optional LangChain message list for multi-turn context.

    Returns:
        dict with at minimum:
            - answer: str
            - sources: list[dict]
            - mode: "router"
            - routed_to: str (human-readable name of the chosen pipeline)
            - classification_reasoning: str (why that pipeline was chosen)
    """
    # Step 1: Classify the question
    classification = classify_query(question)
    category = classification["category"]
    reasoning = classification["reasoning"]

    console.print(
        f"[cyan]Router: classified as '{category}' → "
        f"{CATEGORY_TO_LABEL.get(category, category)}[/cyan]"
    )

    # Step 2: Dispatch to the matching QueryService method
    method_name = CATEGORY_TO_METHOD.get(category, "query_hybrid")
    query_method = getattr(query_service, method_name)

    try:
        result = query_method(question, chat_history=chat_history)
    except Exception as e:
        # If the routed pipeline fails, return an error with routing info
        console.print(f"[red]Router: routed pipeline '{method_name}' failed: {e}[/red]")
        return {
            "answer": f"The router selected {CATEGORY_TO_LABEL.get(category, category)} "
                      f"but it encountered an error: {e}",
            "sources": [],
            "mode": "router",
            "routed_to": CATEGORY_TO_LABEL.get(category, category),
            "classification_reasoning": reasoning,
        }

    # Step 3: Enrich the result with routing metadata
    result["mode"] = "router"
    result["routed_to"] = CATEGORY_TO_LABEL.get(category, category)
    result["classification_reasoning"] = reasoning

    return result
