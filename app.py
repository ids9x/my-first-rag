"""
Gradio web interface for the RAG system.

Launch with:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""
import gradio as gr
from core.query_service import QueryService
from config.settings import WEB_HOST, WEB_PORT, WEB_SHARE, CHAT_MODEL, EMBED_MODEL

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

    # Append reasoning steps for agentic mode
    reasoning_steps = result.get("reasoning_steps", [])
    if reasoning_steps:
        answer += "\n\n<details>\n<summary><b>🧠 Show Reasoning Steps</b></summary>\n\n"
        for step in reasoning_steps:
            if step:  # Skip empty lines
                answer += f"{step}\n\n"
        answer += "</details>"

    return answer


def chat_fn(message: str, history: list, mode: str) -> str:
    """
    Process user message and return response.

    Args:
        message: User's question
        history: Chat history (not used currently)
        mode: Selected query mode

    Returns:
        Formatted response string
    """
    if not message.strip():
        return "Please enter a question."

    # Check system status
    status = query_service.get_status()
    if not status["vector_store_exists"]:
        return "❌ **Vector store not found.**\n\nPlease run:\n```bash\npython -m scripts.ingest\n```"

    try:
        # Route to appropriate query method based on mode
        if mode == "Basic (Vector)":
            result = query_service.query_basic(message)

        elif mode == "Hybrid (Vector+BM25)":
            if not status["bm25_exists"]:
                return "⚠️ **BM25 index not found.**\n\nTo enable hybrid search, run:\n```bash\npython -m scripts.ingest --build-bm25\n```"
            result = query_service.query_hybrid(message)

        elif mode == "Agentic (Multi-step)":
            result = query_service.query_agentic(message)

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
    def respond(message, chat_history, query_mode):
        """Handle user message and stream response token by token."""
        if not message.strip():
            yield chat_history, ""
            return

        chat_history.append({"role": "user", "content": message})

        # Check system status
        status = query_service.get_status()
        if not status["vector_store_exists"]:
            chat_history.append({"role": "assistant", "content": "❌ **Vector store not found.**\n\nPlease run:\n```bash\npython -m scripts.ingest\n```"})
            yield chat_history, ""
            return

        try:
            # Agentic mode: no token streaming (AgentExecutor doesn't support it simply)
            if query_mode == "Agentic (Multi-step)":
                response = chat_fn(message, chat_history, query_mode)
                chat_history.append({"role": "assistant", "content": response})
                yield chat_history, ""
                return

            # Basic & Hybrid: stream tokens
            if query_mode == "Basic (Vector)":
                stream = query_service.query_basic_stream(message)
            elif query_mode == "Hybrid (Vector+BM25)":
                if not status["bm25_exists"]:
                    chat_history.append({"role": "assistant", "content": "⚠️ **BM25 index not found.**\n\nTo enable hybrid search, run:\n```bash\npython -m scripts.ingest --build-bm25\n```"})
                    yield chat_history, ""
                    return
                stream = query_service.query_hybrid_stream(message)
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

    # Create Gradio interface with Blocks for custom layout
    with gr.Blocks(title="RAG System") as demo:
        gr.Markdown("# 🔬 Nuclear RAG System")
        gr.Markdown("Ask questions about the document corpus")

        # Status panel (collapsible)
        with gr.Accordion("📊 System Status", open=False):
            gr.Markdown(status_md)

        # Mode selector
        mode = gr.Radio(
            choices=[
                "Basic (Vector)",
                "Hybrid (Vector+BM25)",
                "Agentic (Multi-step)"
            ],
            value="Basic (Vector)",
            label="Query Mode",
            info="Basic: Semantic search | Hybrid: Vector + keywords | Agentic: Multi-step reasoning"
        )

        # Chat components (manual implementation for proper integration)
        chatbot = gr.Chatbot(label="Chat", height=400)
        msg = gr.Textbox(
            label="Message",
            placeholder="Ask a question about the document corpus...",
            lines=1,
            max_lines=5,
            submit_btn=True
        )

        with gr.Row():
            submit = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")

        # Connect components
        submit.click(respond, [msg, chatbot, mode], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, mode], [chatbot, msg])
        clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

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
        theme=gr.themes.Soft(),
    )
