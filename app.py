"""
Gradio web interface for the RAG system.

Launch with:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""
import tempfile
from datetime import datetime

import gradio as gr
from gradio.themes.utils import fonts
from langchain_core.messages import HumanMessage, AIMessage

from core.query_service import QueryService
from config.settings import (
    WEB_HOST, WEB_PORT, WEB_SHARE, CHAT_MODEL, EMBED_MODEL,
    MULTI_TURN_MAX_EXCHANGES, PROMPT_PRESETS, DEFAULT_PROMPT_PRESET,
)


def build_theme() -> gr.themes.Base:
    """Build a light high-contrast theme with Inter font for readability."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        text_size=gr.themes.sizes.text_lg,
        font=[
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    ).set(
        # --- Light background ---
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        background_fill_primary="#f8f9fa",
        background_fill_primary_dark="#f8f9fa",
        background_fill_secondary="#e9ecef",
        background_fill_secondary_dark="#e9ecef",
        # --- High-contrast text ---
        body_text_color="#1a1a2e",
        body_text_color_dark="#1a1a2e",
        body_text_color_subdued="#495057",
        body_text_color_subdued_dark="#495057",
        # --- Borders and inputs ---
        border_color_primary="#ced4da",
        border_color_primary_dark="#ced4da",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#adb5bd",
        input_border_color_dark="#adb5bd",
        # --- Buttons ---
        button_primary_background_fill="#0066cc",
        button_primary_background_fill_dark="#0066cc",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#e9ecef",
        button_secondary_background_fill_dark="#e9ecef",
        button_secondary_text_color="#1a1a2e",
        button_secondary_text_color_dark="#1a1a2e",
        # --- Blocks (panels, accordions) ---
        block_background_fill="#f8f9fa",
        block_background_fill_dark="#f8f9fa",
        block_border_color="#dee2e6",
        block_border_color_dark="#dee2e6",
        block_label_text_color="#1a1a2e",
        block_label_text_color_dark="#1a1a2e",
        block_title_text_color="#1a1a2e",
        block_title_text_color_dark="#1a1a2e",
    )

# Initialize query service globally
query_service = QueryService()


def format_response(result: dict) -> str:
    """
    Format service response for display in chat interface.

    Args:
        result: Dict from query_service with answer, sources, etc.

    Returns:
        Formatted markdown string
    """
    answer = result["answer"]

    # Append sources if available
    sources = result.get("sources", [])
    if sources:
        answer += "\n\n---\n\n**📚 Sources:**\n"
        for i, src in enumerate(sources, 1):
            answer += f"{i}. {src['source']}, page {src['page']}\n"

        # Add collapsible source excerpts for verification
        previews = [s for s in sources if s.get("content_preview")]
        if previews:
            answer += "\n<details>\n<summary><b>📄 Source Excerpts</b></summary>\n\n"
            for i, src in enumerate(previews, 1):
                answer += f"**{i}. {src['source']}, p.{src['page']}**\n\n"
                answer += f"> {src['content_preview']}\n\n"
            answer += "</details>"

    # Append reasoning steps for agentic mode
    reasoning_steps = result.get("reasoning_steps", [])
    if reasoning_steps:
        answer += "\n\n<details>\n<summary><b>🧠 Show Reasoning Steps</b></summary>\n\n"
        for step in reasoning_steps:
            if step:  # Skip empty lines
                answer += f"{step}\n\n"
        answer += "</details>"

    # Append map summaries for map-reduce mode
    map_summaries = result.get("map_summaries", [])
    if map_summaries:
        chunk_count = result.get("chunk_count", len(map_summaries))
        answer += (
            f"\n\n<details>\n<summary><b>🗺️ Map Summaries "
            f"({chunk_count} chunks)</b></summary>\n\n"
        )
        for summary in map_summaries:
            answer += f"- {summary}\n\n"
        answer += "</details>"

    # Append strategy contributions for parallel mode
    strategy_counts = result.get("strategy_counts")
    if strategy_counts:
        total = result.get("total_unique_chunks", 0)
        answer += (
            f"\n\n<details>\n<summary><b>⚡ Strategy Contributions "
            f"({total} unique chunks)</b></summary>\n\n"
        )
        for strategy, count in strategy_counts.items():
            answer += f"- **{strategy}**: {count} chunks\n\n"
        answer += "</details>"

    # Append routing decision for router mode
    routed_to = result.get("routed_to")
    if routed_to:
        reasoning = result.get("classification_reasoning", "")
        answer += "\n\n<details>\n<summary><b>🔀 Routing Decision</b></summary>\n\n"
        answer += f"**Routed to:** {routed_to}\n\n"
        answer += f"**Reasoning:** {reasoning}\n\n"
        answer += "</details>"

    return answer


def _thinking_message(query_mode: str) -> str:
    """Return a mode-specific thinking/progress message."""
    messages = {
        "Router (Auto)": "Classifying question and routing to best pipeline...",
        "Agentic (Multi-step)": "Reasoning through multiple retrieval steps...",
        "Map-Reduce": "Retrieving chunks and processing each independently...",
        "Parallel + Merge": "Running multiple retrieval strategies in parallel...",
    }
    return f"**{messages.get(query_mode, 'Processing...')}**"


def _build_chat_history(chat_history: list, max_exchanges: int) -> list:
    """Convert recent Gradio chat messages to LangChain message format.

    Takes the most recent exchanges (excluding the current user message,
    which is the last item) and converts them to HumanMessage/AIMessage.

    Args:
        chat_history: Gradio chat list [{"role": "user"|"assistant", "content": ...}]
        max_exchanges: Maximum number of Q&A pairs to include

    Returns:
        List of LangChain HumanMessage/AIMessage objects
    """
    # Exclude the latest user message (it's passed as the question directly)
    prior = chat_history[:-1]
    # Take last N exchanges (2 messages per exchange)
    prior = prior[-(max_exchanges * 2):]
    messages = []
    for msg in prior:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def export_chat(chat_history: list):
    """Export chat history as a downloadable markdown file.

    Args:
        chat_history: Gradio chat list [{"role": "user"|"assistant", "content": ...}]

    Returns:
        gr.File update with the exported file, or hidden if chat is empty
    """
    if not chat_history:
        return gr.File(visible=False)

    lines = [
        f"# RAG Chat Export",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Messages:** {len(chat_history)}",
        "",
        "---",
        "",
    ]
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Gradio may store content as a list of parts — flatten to string
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        lines.append(f"### {role}")
        lines.append("")
        lines.append(str(content))
        lines.append("")
        lines.append("---")
        lines.append("")

    md_content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="rag_chat_", delete=False,
    )
    tmp.write(md_content)
    tmp.close()
    return gr.File(value=tmp.name, visible=True)


def chat_fn(message: str, history: list, mode: str,
            chat_history_lc: list | None = None,
            system_prompt: str | None = None) -> str:
    """
    Process user message and return response.

    Args:
        message: User's question
        history: Chat history (Gradio format, for context)
        mode: Selected query mode
        chat_history_lc: Optional LangChain message list for multi-turn context
        system_prompt: Optional custom system prompt text

    Returns:
        Formatted response string
    """
    if not message.strip():
        return "Please enter a question."

    # Normalize empty prompt to None (use default)
    if system_prompt is not None and not system_prompt.strip():
        system_prompt = None

    # Check system status
    status = query_service.get_status()
    if not status["vector_store_exists"]:
        return "❌ **Vector store not found.**\n\nPlease run:\n```bash\npython -m scripts.ingest\n```"

    try:
        # Route to appropriate query method based on mode
        if mode == "Basic (Vector)":
            result = query_service.query_basic(message, chat_history=chat_history_lc,
                                               system_prompt=system_prompt)

        elif mode == "Hybrid (Vector+BM25)":
            if not status["bm25_exists"]:
                return "⚠️ **BM25 index not found.**\n\nTo enable hybrid search, run:\n```bash\npython -m scripts.ingest --build-bm25\n```"
            result = query_service.query_hybrid(message, chat_history=chat_history_lc,
                                                system_prompt=system_prompt)

        elif mode == "Agentic (Multi-step)":
            result = query_service.query_agentic(message, chat_history=chat_history_lc,
                                                 system_prompt=system_prompt)

        elif mode == "Router (Auto)":
            result = query_service.query_router(message, chat_history=chat_history_lc,
                                                system_prompt=system_prompt)

        elif mode == "Map-Reduce":
            result = query_service.query_map_reduce(message, system_prompt=system_prompt)

        elif mode == "Parallel + Merge":
            result = query_service.query_parallel(message, chat_history=chat_history_lc,
                                                  system_prompt=system_prompt)

        else:
            return f"❌ Unknown mode: {mode}"

        return format_response(result)

    except Exception as e:
        error_msg = str(e)

        # Provide helpful error messages
        if "connection refused" in error_msg.lower() or "connect" in error_msg.lower():
            return "❌ **Cannot connect to llama-server.**\n\nPlease ensure llama-server is running:\n```bash\n/home/ids9x/start-llama-server.sh\n```"
        else:
            return f"❌ **Error:** {error_msg}"


def create_interface():
    """Create and configure the Gradio interface."""

    # Get system status for display
    status = query_service.get_status()

    # Build status message
    status_lines = []
    status_lines.append(f"**Vector Store:** {'✅' if status['vector_store_exists'] else '❌'} "
                       f"({status['vector_store_count']} chunks)")
    status_lines.append(f"**BM25 Index:** {'✅' if status['bm25_exists'] else '❌'}")
    status_lines.append(f"**Reranker:** {'✅ Enabled' if status['reranker_enabled'] else '❌ Disabled'}")
    status_lines.append(f"**Chat Model:** `{CHAT_MODEL}`")
    status_lines.append(f"**Embedding Model:** `{EMBED_MODEL}`")
    status_md = "\n\n".join(status_lines)

    # Helper function for chat interactions (streaming)
    def respond(message, chat_history, query_mode, multi_turn, system_prompt_text):
        """Handle user message and stream response token by token."""
        if not message.strip():
            yield chat_history, ""
            return

        chat_history.append({"role": "user", "content": message})

        # Normalize system prompt (empty string → None → use default)
        sys_prompt = system_prompt_text.strip() if system_prompt_text else None
        if not sys_prompt:
            sys_prompt = None

        # Build LangChain chat history for multi-turn context
        chat_history_lc = None
        if multi_turn:
            chat_history_lc = _build_chat_history(
                chat_history, MULTI_TURN_MAX_EXCHANGES,
            )

        # Check system status
        status = query_service.get_status()
        if not status["vector_store_exists"]:
            chat_history.append({"role": "assistant", "content": "❌ **Vector store not found.**\n\nPlease run:\n```bash\npython -m scripts.ingest\n```"})
            yield chat_history, ""
            return

        try:
            # Agentic, Router, Map-Reduce, Parallel: no token streaming
            # Show a thinking message while processing
            if query_mode in ("Agentic (Multi-step)", "Router (Auto)", "Map-Reduce", "Parallel + Merge"):
                chat_history.append({"role": "assistant", "content": _thinking_message(query_mode)})
                yield chat_history, ""

                response = chat_fn(message, chat_history, query_mode,
                                   chat_history_lc, system_prompt=sys_prompt)
                chat_history[-1]["content"] = response
                yield chat_history, ""
                return

            # Basic & Hybrid: stream tokens
            if query_mode == "Basic (Vector)":
                stream = query_service.query_basic_stream(
                    message, chat_history=chat_history_lc,
                    system_prompt=sys_prompt,
                )
            elif query_mode == "Hybrid (Vector+BM25)":
                if not status["bm25_exists"]:
                    chat_history.append({"role": "assistant", "content": "⚠️ **BM25 index not found.**\n\nTo enable hybrid search, run:\n```bash\npython -m scripts.ingest --build-bm25\n```"})
                    yield chat_history, ""
                    return
                stream = query_service.query_hybrid_stream(
                    message, chat_history=chat_history_lc,
                    system_prompt=sys_prompt,
                )
            else:
                chat_history.append({"role": "assistant", "content": f"❌ Unknown mode: {query_mode}"})
                yield chat_history, ""
                return

            # Add empty assistant message that we'll update as tokens arrive
            chat_history.append({"role": "assistant", "content": ""})

            for chunk in stream:
                if "partial" in chunk:
                    # Update the last message with streamed tokens
                    chat_history[-1]["content"] = chunk["partial"]
                    yield chat_history, ""
                elif "answer" in chunk:
                    # Final chunk: format with sources
                    chat_history[-1]["content"] = format_response(chunk)
                    yield chat_history, ""

        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower() or "connect" in error_msg.lower():
                msg = "❌ **Cannot connect to llama-server.**\n\nPlease ensure llama-server is running:\n```bash\n/home/ids9x/start-llama-server.sh\n```"
            else:
                msg = f"❌ **Error:** {error_msg}"
            chat_history.append({"role": "assistant", "content": msg})
            yield chat_history, ""

    # JavaScript to update font size on the chat and message areas dynamically.
    # Targets the chatbot messages and input textbox via their elem_id.
    font_size_js = """
    (size) => {
        const px = size + 'px';
        const style = document.getElementById('dynamic-font-style');
        if (style) {
            style.textContent = `
                #rag-chatbot { --chatbot-text-size: ${px} !important; }
                #rag-chatbot .message-wrap { font-size: ${px} !important; }
                #rag-chatbot .message .prose { font-size: ${px} !important; }
                #rag-chatbot .message .prose * { font-size: inherit !important; }
                #rag-msg textarea { font-size: ${px} !important; }
            `;
        }
        return size;
    }
    """

    # Create Gradio interface with Blocks for custom layout
    with gr.Blocks(
        title="RAG System",
        css="""
            #rag-chatbot { --chatbot-text-size: 16px; }
            #rag-chatbot .message-wrap { font-size: 16px; }
            #rag-chatbot .message .prose { font-size: 16px; }
            #rag-chatbot .message .prose * { font-size: inherit; }
            #rag-msg textarea { font-size: 16px; }
        """,
        head='<style id="dynamic-font-style"></style>',
    ) as demo:
        gr.Markdown("# 🔬 RAG System")
        gr.Markdown("Ask questions about the document corpus")

        # Status panel (collapsible)
        with gr.Accordion("📊 System Status", open=False):
            gr.Markdown(status_md)

        # Mode selector
        mode = gr.Radio(
            choices=[
                "Basic (Vector)",
                "Hybrid (Vector+BM25)",
                "Agentic (Multi-step)",
                "Router (Auto)",
                "Map-Reduce",
                "Parallel + Merge",
            ],
            value="Basic (Vector)",
            label="Query Mode",
            info="Basic: Semantic search | Hybrid: Vector + keywords | Agentic: Multi-step | Router: Auto-selects | Map-Reduce: Per-chunk analysis | Parallel: Multi-strategy"
        )

        # System prompt preset selector + custom editor
        with gr.Accordion("🎯 System Prompt", open=False):
            preset_names = list(PROMPT_PRESETS.keys())
            prompt_preset = gr.Dropdown(
                choices=preset_names,
                value=DEFAULT_PROMPT_PRESET,
                label="Preset",
                info="Select a preset or edit the text below to customise",
            )
            system_prompt_text = gr.Textbox(
                label="Active System Prompt",
                value=PROMPT_PRESETS[DEFAULT_PROMPT_PRESET],
                lines=5,
                max_lines=10,
                info="Edit this text to change how the LLM responds. "
                     "Selecting a preset above will replace this text.",
            )

            def on_preset_change(preset_name):
                """When a preset is selected, populate the text area."""
                return PROMPT_PRESETS.get(preset_name, "")

            prompt_preset.change(
                fn=on_preset_change,
                inputs=prompt_preset,
                outputs=system_prompt_text,
            )

        # FAQ: when to use each mode (collapsible)
        with gr.Accordion("❓ Which Query Mode Should I Use?", open=False):
            gr.Markdown(
                "### Basic (Vector)\n"
                "**Best for:** Quick factual lookups, definitions, and single-concept questions.\n\n"
                "Uses semantic similarity to find the most relevant chunks. Fastest mode — "
                "start here if you're unsure.\n\n"
                "- *\"What is the definition of significant environmental impact?\"*\n"
                "- *\"What does Section 3.2 say about air quality monitoring?\"*\n\n"
                "---\n"
                "### Hybrid (Vector + BM25)\n"
                "**Best for:** Questions with specific terminology, section numbers, "
                "or technical terms that need exact keyword matching.\n\n"
                "Combines semantic search with keyword search (BM25) using Reciprocal "
                "Rank Fusion. Catches both paraphrased concepts and exact terms.\n\n"
                "- *\"What mitigation measures does EIA Section 4.5.2 require for noise?\"*\n"
                "- *\"Compare PM2.5 and PM10 emission limits across the reports\"*\n\n"
                "---\n"
                "### Agentic (Multi-step)\n"
                "**Best for:** Complex questions that require reasoning across multiple "
                "sections or documents — the LLM decides which tools to use.\n\n"
                "The agent can perform multiple retrieval rounds, refine its search, "
                "and chain reasoning steps. Shows its thought process.\n\n"
                "- *\"What are the cumulative impacts of the project and how do the "
                "proposed mitigations address each one?\"*\n"
                "- *\"Trace the relationship between the biodiversity baseline and "
                "the predicted residual impacts\"*\n\n"
                "---\n"
                "### Router (Auto)\n"
                "**Best for:** When you're not sure which mode to pick — let the LLM "
                "classify your question and route it automatically.\n\n"
                "A lightweight LLM call classifies your question as factual, "
                "comparative, or exploratory, then dispatches to Basic, Hybrid, or "
                "Agentic accordingly. Adds ~2 seconds overhead.\n\n"
                "- Any question works — the router decides the best pipeline.\n\n"
                "---\n"
                "### Map-Reduce\n"
                "**Best for:** Thorough analysis where you don't want the LLM to skip "
                "details buried in the middle of long content.\n\n"
                "Retrieves 8 chunks, processes each independently (map phase), then "
                "synthesises all summaries into a final answer (reduce phase). "
                "Slowest mode but most thorough.\n\n"
                "- *\"List all environmental receptors identified across all assessment "
                "chapters\"*\n"
                "- *\"What are every monitoring requirement specified in the reports?\"*\n\n"
                "---\n"
                "### Parallel + Merge\n"
                "**Best for:** Broadest possible coverage — runs semantic, keyword, and "
                "precision retrieval concurrently then merges the results.\n\n"
                "Documents found by multiple strategies rank higher. Good when you want "
                "to be confident nothing relevant was missed.\n\n"
                "- *\"Explain the assessment methodology used for water quality impacts\"*\n"
                "- *\"What are the key findings regarding habitat loss and species "
                "displacement?\"*\n"
            )

        # Chat components
        chatbot = gr.Chatbot(label="Chat", height=500, elem_id="rag-chatbot")
        msg = gr.Textbox(
            label="Message",
            placeholder="Ask a question about the document corpus...",
            lines=1,
            max_lines=5,
            submit_btn=True,
            elem_id="rag-msg",
        )

        with gr.Row():
            submit = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")
            export_btn = gr.Button("Export Chat")

        multi_turn = gr.Checkbox(
            label="Multi-turn context",
            value=False,
            info="Pass recent chat history to the LLM for follow-up questions",
        )

        export_file = gr.File(label="Exported Chat", visible=False)

        # Example questions the user can click to auto-fill
        gr.Examples(
            examples=[
                "What is the definition of significant environmental impact?",
                "Compare the air quality and noise mitigation measures",
                "List all monitoring requirements specified in the reports",
                "What are the key findings regarding water quality impacts?",
            ],
            inputs=msg,
            label="Example Questions",
        )

        # Display settings (collapsible)
        with gr.Accordion("Display Settings", open=False):
            font_slider = gr.Slider(
                minimum=12,
                maximum=24,
                value=16,
                step=1,
                label="Chat Font Size (px)",
                info="Adjust the text size in the chat and message areas",
            )
            font_slider.change(fn=None, inputs=font_slider, outputs=font_slider,
                               js=font_size_js)

        # Connect components
        submit.click(respond, [msg, chatbot, mode, multi_turn, system_prompt_text], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, mode, multi_turn, system_prompt_text], [chatbot, msg])
        clear.click(
            lambda: ([], "", gr.File(visible=False)),
            None, [chatbot, msg, export_file], queue=False,
        )
        export_btn.click(export_chat, [chatbot], [export_file])

    return demo


if __name__ == "__main__":
    demo = create_interface()

    print("\n" + "="*60)
    print("🚀 Starting Gradio web interface...")
    print(f"📍 URL: http://{WEB_HOST}:{WEB_PORT}")
    print("="*60 + "\n")

    demo.launch(
        server_name=WEB_HOST,
        server_port=WEB_PORT,
        share=WEB_SHARE,
        theme=build_theme(),
    )
