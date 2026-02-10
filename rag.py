"""
My First RAG — Ask questions about a local PDF using Ollama.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ── Configuration ──────────────────────────────────────────────
CHAT_MODEL = "gemma3:4b"           # The model you pulled in Step 13
EMBED_MODEL = "nomic-embed-text"   # The embedding model from Step 1
PDF_PATH = "/home/ids9x/my-first-rag/data/ASME-NQA-1-2022-Proofs-extract.pdf"     # Path to your PDF (change this)
CHROMA_DIR = "./chroma_db"         # Where the vector database is stored on disk
# ───────────────────────────────────────────────────────────────


def load_and_split_pdf(pdf_path: str) -> list:
    """Load a PDF and split it into chunks."""

    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   Loaded {len(pages)} pages.")

    # Split into chunks. Each chunk is ~1000 characters with 200 characters
    # of overlap so that context isn't lost at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
    print(f"   Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Embed the chunks and store them in ChromaDB."""

    print("🔢 Creating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"   Stored {len(chunks)} chunks in {CHROMA_DIR}")
    return vector_store


def build_rag_chain(vector_store: Chroma):
    """Build the RAG chain: retrieve context → prompt the LLM → parse the answer."""

    # The retriever searches ChromaDB for the 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # The prompt template tells the LLM how to use the retrieved context
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based ONLY on the
following context. If the context does not contain enough information to
answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    )

    # The chat model
    llm = ChatOllama(model=CHAT_MODEL)

    # Helper function to format retrieved documents into a single string
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Wire it all together into a chain:
    # 1. Retrieve relevant chunks and format them as "context"
    # 2. Pass the original question through as "question"
    # 3. Feed both into the prompt template
    # 4. Send the prompt to the LLM
    # 5. Parse the output as a string
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    # Step A: Load and chunk the PDF
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at '{PDF_PATH}'. Place your PDF in the data/ folder")
        print(f"   and update PDF_PATH at the top of this script.")
        return

    chunks = load_and_split_pdf(PDF_PATH)

    # Step B: Create (or recreate) the vector store
    vector_store = create_vector_store(chunks)

    # Step C: Build the RAG chain
    chain = build_rag_chain(vector_store)

    # Step D: Interactive question loop
    print("\n✅ RAG is ready! Ask questions about your document.")
    print("   Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("Thinking...")
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()