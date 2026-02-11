"""Nuclear domain metadata extraction from chunk content and context."""
import re

# Standard identifiers to detect in chunk text or section paths
STANDARD_PATTERNS = {
    "NQA-1": r"NQA[-\s]?1",
    "ASME": r"ASME",
    "10CFR50": r"10\s?CFR\s?50",
    "10CFR21": r"10\s?CFR\s?21",
    "IAEA": r"IAEA",
    "SSR": r"SSR[-\s]?\d",
    "RCC-M": r"RCC[-\s]?M",
}

DOC_TYPE_KEYWORDS = {
    "requirement": ["shall", "must", "required", "mandatory"],
    "guidance": ["should", "recommended", "may consider", "guidance"],
    "informative": ["for information", "annex", "appendix", "note"],
    "definition": ["means", "is defined as", "refers to"],
}


def enrich_metadata(chunk_text: str, metadata: dict) -> dict:
    """Add nuclear-domain tags to a chunk's metadata dict."""
    combined = f"{metadata.get('section_path', '')} {chunk_text}".lower()

    # Detect which standards are referenced
    standards_found = []
    for name, pattern in STANDARD_PATTERNS.items():
        if re.search(pattern, combined, re.IGNORECASE):
            standards_found.append(name)
    metadata["standards_referenced"] = ",".join(standards_found) if standards_found else ""

    # Classify document type by keyword density
    doc_type = "general"
    max_hits = 0
    for dtype, keywords in DOC_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > max_hits:
            max_hits = hits
            doc_type = dtype
    metadata["doc_type"] = doc_type

    # Detect cross-references (e.g. "Section 3.2.1", "per Requirement 7")
    xrefs = re.findall(
        r"(?:Section|Clause|Part|Requirement|Subpart|Appendix|Annex)\s+[\d.]+",
        chunk_text,
        re.IGNORECASE,
    )
    metadata["cross_references"] = ",".join(xrefs) if xrefs else ""

    return metadata