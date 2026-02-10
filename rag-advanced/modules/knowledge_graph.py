"""
Knowledge Graph Extraction (Priority 6 — txt2kg)

Extracts structured (subject, predicate, object) triples from document
chunks using the LLM, then stores them in a NetworkX graph.

Use cases for nuclear docs:
  - "Which standards reference NQA-1?"
  - "What are the requirements for QA Level 1 suppliers?"
  - "Show me the relationship between ASME Section III and 10 CFR 50"

This is the most experimental module. Works best with larger models
(qwen2.5:32b+) that can follow extraction instructions reliably.
"""
import json
from pathlib import Path
import networkx as nx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm
from config.settings import KNOWLEDGE_GRAPH_DIR, KG_MAX_TRIPLES_PER_CHUNK


EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Extract factual relationships from the following text as a JSON list of triples.
Each triple should have: "subject", "predicate", "object".

Focus on:
- Standards and their requirements (e.g., "NQA-1" → "requires" → "document control")
- Organizational relationships (e.g., "IAEA" → "publishes" → "Safety Guide NS-G-1.1")
- Regulatory references (e.g., "10 CFR 50 Appendix B" → "references" → "NQA-1")
- Component classifications (e.g., "Safety Class 1" → "includes" → "reactor coolant pressure boundary")

Return ONLY a JSON array. No markdown, no explanation. Example format:
[
  {{"subject": "NQA-1", "predicate": "requires", "object": "document control"}},
  {{"subject": "ASME Section III", "predicate": "covers", "object": "nuclear power plant components"}}
]

If no clear relationships can be extracted, return an empty array: []

Text:
{text}

Triples (JSON only):"""
)


class KnowledgeGraph:
    """
    Builds and queries a knowledge graph from document chunks.
    """

    def __init__(self, graph_dir: str | Path = KNOWLEDGE_GRAPH_DIR):
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.graph_dir / "knowledge_graph.graphml"
        self.graph = self._load_or_create()

    def _load_or_create(self) -> nx.DiGraph:
        """Load existing graph or create a new one."""
        if self.graph_path.exists():
            print(f"📂 Loading knowledge graph from {self.graph_path}")
            g = nx.read_graphml(self.graph_path)
            print(f"   {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
            return g
        print("🆕 Creating new knowledge graph")
        return nx.DiGraph()

    def save(self):
        """Persist graph to disk."""
        nx.write_graphml(self.graph, self.graph_path)
        print(f"💾 Graph saved: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def extract_from_chunks(
        self,
        chunks: list[Document],
        max_triples: int = KG_MAX_TRIPLES_PER_CHUNK,
    ):
        """
        Use the LLM to extract triples from each chunk.
        This is slow — call it during ingestion, not at query time.
        """
        llm = get_llm(temperature=0.0)
        chain = EXTRACTION_PROMPT | llm | StrOutputParser()

        total_triples = 0
        for i, chunk in enumerate(chunks):
            print(f"   Extracting from chunk {i + 1}/{len(chunks)}...", end=" ")
            try:
                raw = chain.invoke({"text": chunk.page_content})
                # Clean potential markdown fences
                raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
                triples = json.loads(raw)

                if not isinstance(triples, list):
                    print("⚠️ Not a list, skipping.")
                    continue

                count = 0
                for triple in triples[:max_triples]:
                    s = triple.get("subject", "").strip()
                    p = triple.get("predicate", "").strip()
                    o = triple.get("object", "").strip()
                    if s and p and o:
                        self.graph.add_edge(s, o, relation=p)
                        count += 1

                print(f"{count} triples")
                total_triples += count

            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Error: {e}")

        print(f"\n📊 Extracted {total_triples} triples total.")
        self.save()

    def query_entity(self, entity: str) -> dict:
        """Find all relationships for an entity."""
        entity_lower = entity.lower()

        # Find matching nodes (case-insensitive)
        matches = [n for n in self.graph.nodes() if entity_lower in n.lower()]

        results = {"outgoing": [], "incoming": []}
        for node in matches:
            for _, target, data in self.graph.out_edges(node, data=True):
                results["outgoing"].append({
                    "subject": node,
                    "predicate": data.get("relation", "related_to"),
                    "object": target,
                })
            for source, _, data in self.graph.in_edges(node, data=True):
                results["incoming"].append({
                    "subject": source,
                    "predicate": data.get("relation", "related_to"),
                    "object": node,
                })

        return results

    def format_query_results(self, entity: str) -> str:
        """Human-readable query results for an entity."""
        results = self.query_entity(entity)

        if not results["outgoing"] and not results["incoming"]:
            return f"No relationships found for '{entity}'."

        lines = [f"Knowledge graph results for '{entity}':\n"]

        if results["outgoing"]:
            lines.append("Outgoing relationships:")
            for r in results["outgoing"]:
                lines.append(f"  {r['subject']} → {r['predicate']} → {r['object']}")

        if results["incoming"]:
            lines.append("\nIncoming relationships:")
            for r in results["incoming"]:
                lines.append(f"  {r['subject']} → {r['predicate']} → {r['object']}")

        return "\n".join(lines)

    def get_stats(self) -> str:
        """Return graph statistics."""
        g = self.graph
        return (
            f"Nodes: {g.number_of_nodes()}, "
            f"Edges: {g.number_of_edges()}, "
            f"Components: {nx.number_weakly_connected_components(g)}"
        )
