"""
Multilingual RAG (Priority 4)

Handles German and English documents in the same vector store.
Key features:
  - Language detection on queries
  - Bilingual prompt templates (answers in the query's language)
  - Metadata tagging for language filtering

nomic-embed-text already handles multilingual content reasonably well,
so we don't need separate embedding models. The main gain here is in
the prompt engineering and retrieval filtering.

For later (when you scale up):
  - Consider multilingual-e5-large for better cross-lingual retrieval
  - Pull via: ollama pull jeffh/intfloat-multilingual-e5-large:f16
"""
from langdetect import detect, DetectorFactory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Make language detection deterministic
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """
    Detect language of text. Returns 'de' or 'en' (defaults to 'en').
    """
    try:
        lang = detect(text)
        return "de" if lang == "de" else "en"
    except Exception:
        return "en"


def tag_document_language(doc: Document) -> Document:
    """Add language metadata to a document chunk."""
    lang = detect_language(doc.page_content)
    doc.metadata["language"] = lang
    return doc


def tag_documents_language(docs: list[Document]) -> list[Document]:
    """Tag all documents with their detected language."""
    tagged = [tag_document_language(doc) for doc in docs]

    lang_counts = {}
    for doc in tagged:
        lang = doc.metadata.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"🌐 Language distribution: {lang_counts}")
    return tagged


# ── Prompt Templates ───────────────────────────────────────────

PROMPT_EN = ChatPromptTemplate.from_template(
    """You are a helpful assistant specializing in nuclear regulatory documents.
Answer the question based ONLY on the following context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
)

PROMPT_DE = ChatPromptTemplate.from_template(
    """Du bist ein hilfreicher Assistent für nukleare Regulierungsdokumente.
Beantworte die Frage NUR auf Basis des folgenden Kontexts.
Wenn der Kontext nicht genug Information enthält, sage das.

Kontext:
{context}

Frage: {question}

Antwort:"""
)

PROMPT_BILINGUAL = ChatPromptTemplate.from_template(
    """You are a helpful assistant specializing in nuclear regulatory documents.
You work with documents in both German and English.
Answer the question based ONLY on the following context.
Respond in the same language as the question.
If the context does not contain enough information, say so.

Context (may contain both German and English passages):
{context}

Question: {question}

Answer:"""
)


def get_prompt_for_language(lang: str = "en") -> ChatPromptTemplate:
    """Return the appropriate prompt template for the detected language."""
    if lang == "de":
        return PROMPT_DE
    return PROMPT_EN


def get_bilingual_prompt() -> ChatPromptTemplate:
    """Return the bilingual prompt that handles mixed-language context."""
    return PROMPT_BILINGUAL
