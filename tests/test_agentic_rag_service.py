import unittest

from src.services.agentic_rag.config import AgenticRAGConfig
from src.services.agentic_rag.service import AgenticRAGService
from src.services.opensearch.hybrid_service import ChunkHit, HybridSearchResult
from src.services.rag.context_builder import ContextBuilder


def _transformer_hit() -> ChunkHit:
    return ChunkHit(
        arxiv_id="1706.03762",
        chunk_id="chunk-1",
        chunk_text="Section: Model Architecture\nThe Transformer uses self-attention.",
        section_name="Model Architecture",
        chunk_index=0,
        title="Attention Is All You Need",
        abstract="Transformer architecture paper.",
        authors=["Vaswani et al."],
        categories=["cs.CL"],
        published_at="2017-06-12",
        score=0.9,
    )


class FakeSearchService:
    def __init__(self, hits: list[ChunkHit]):
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, use_hybrid: bool = True, page_size: int = 5):
        self.queries.append(query)
        return HybridSearchResult(
            query=query,
            hits=self.hits,
            total=len(self.hits),
            took_ms=7,
            search_mode="hybrid" if use_hybrid else "bm25",
            page_size=page_size,
        )


class FakeOllamaService:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        num_predict: int = 512,
        json_mode: bool = False,
    ) -> str:
        if not self.responses:
            raise AssertionError("No fake Ollama response left")
        return self.responses.pop(0)


def _service(search: FakeSearchService, ollama: FakeOllamaService) -> AgenticRAGService:
    return AgenticRAGService(
        search_svc=search,
        ollama_svc=ollama,
        context_builder=ContextBuilder(),
        config=AgenticRAGConfig(max_retrieval_attempts=2, top_k=5),
    )


class AgenticRAGServiceTests(unittest.TestCase):
    def test_out_of_scope_question_stops_before_retrieval(self):
        search = FakeSearchService([_transformer_hit()])
        ollama = FakeOllamaService([
            '{"score": 0, "reason": "Outside CS/AI/ML research"}',
        ])

        result = _service(search, ollama).ask("What is photosynthesis?")

        self.assertEqual(result.retrieval_attempts, 0)
        self.assertEqual(result.guardrail_score, 0)
        self.assertEqual(result.sources, [])
        self.assertEqual(search.queries, [])
        self.assertEqual(
            [step.step for step in result.reasoning_steps],
            ["guardrail", "out_of_scope"],
        )

    def test_relevant_documents_generate_answer(self):
        search = FakeSearchService([_transformer_hit()])
        ollama = FakeOllamaService([
            '{"score": 95, "reason": "Question is about ML architecture"}',
            '{"is_relevant": true, "score": 0.9, "reason": "Directly relevant"}',
            "Transformers use self-attention. [1]",
        ])

        result = _service(search, ollama).ask("What is the transformer architecture?")

        self.assertEqual(result.answer, "Transformers use self-attention. [1]")
        self.assertEqual(result.retrieval_attempts, 1)
        self.assertEqual(result.guardrail_score, 95)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].arxiv_id, "1706.03762")
        self.assertEqual(
            [step.step for step in result.reasoning_steps],
            ["guardrail", "retrieve", "grade_documents", "generate_answer"],
        )

    def test_rewrite_then_insufficient_evidence_after_retry_limit(self):
        search = FakeSearchService([_transformer_hit()])
        ollama = FakeOllamaService([
            '{"score": 70, "reason": "Broad but in-domain"}',
            '{"is_relevant": false, "score": 0.1, "reason": "Weak evidence"}',
            '{"rewritten_query": "specific neural network research", "reason": "More specific"}',
            '{"is_relevant": false, "score": 0.1, "reason": "Still weak"}',
        ])

        result = _service(search, ollama).ask("Tell me about ML stuff")

        self.assertEqual(result.retrieval_attempts, 2)
        self.assertEqual(result.rewritten_query, "specific neural network research")
        self.assertEqual(search.queries, ["Tell me about ML stuff", "specific neural network research"])
        self.assertIn("could not find enough relevant evidence", result.answer)
        self.assertEqual(
            [step.step for step in result.reasoning_steps],
            [
                "guardrail",
                "retrieve",
                "grade_documents",
                "rewrite_query",
                "retrieve",
                "grade_documents",
                "insufficient_evidence",
            ],
        )


if __name__ == "__main__":
    unittest.main()
