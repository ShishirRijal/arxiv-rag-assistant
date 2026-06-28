"""Service orchestration for agentic RAG."""

import logging
import time

from langgraph.graph import END, START, StateGraph

from ..ollama.service import OllamaService
from ..opensearch.hybrid_service import HybridSearchService
from ..rag.context_builder import ContextBuilder
from .config import AgenticRAGConfig
from .models import AgenticRAGResult, ReasoningStep, SourceItem
from .nodes.generate_answer import run_generate_answer_node
from .nodes.grade_documents import route_after_grading, run_grade_documents_node
from .nodes.guardrail import route_after_guardrail, run_guardrail_node
from .nodes.insufficient_evidence import run_insufficient_evidence_node
from .nodes.out_of_scope import run_out_of_scope_node
from .nodes.retrieve import run_retrieve_node
from .nodes.rewrite_query import run_rewrite_query_node
from .state import AgenticRAGState, initial_state

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Graph-controlled RAG service.

    This service keeps the public behavior simple while the internal graph
    decides whether to reject, retrieve, grade, rewrite, retry, or generate.
    """

    def __init__(
        self,
        *,
        search_svc: HybridSearchService,
        ollama_svc: OllamaService,
        context_builder: ContextBuilder,
        config: AgenticRAGConfig,
    ):
        self._search = search_svc
        self._ollama = ollama_svc
        self._context = context_builder
        self._config = config
        self._graph = self._build_graph()

    def ask(self, question: str) -> AgenticRAGResult:
        """Run one question through the agentic RAG graph."""
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty")

        start = time.perf_counter()
        state = initial_state(
            cleaned,
            metadata={
                "model": self._config.model,
                "top_k": self._config.top_k,
                "use_hybrid": self._config.use_hybrid,
            },
        )
        logger.info("Agentic RAG ask: question=%r", cleaned)

        final_state = self._graph.invoke(state)
        took_ms = int((time.perf_counter() - start) * 1000)

        result = self._to_result(final_state, took_ms=took_ms)
        logger.info(
            "Agentic RAG complete: attempts=%d sources=%d took=%dms",
            result.retrieval_attempts,
            len(result.sources),
            result.took_ms,
        )
        return result

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("guardrail", self._guardrail)
        workflow.add_node("out_of_scope", run_out_of_scope_node)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("insufficient_evidence", run_insufficient_evidence_node)

        workflow.add_edge(START, "guardrail")
        workflow.add_conditional_edges(
            "guardrail",
            self._route_after_guardrail,
            {
                "retrieve": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_edge("out_of_scope", END)

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "insufficient_evidence": "insufficient_evidence",
            },
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("insufficient_evidence", END)

        return workflow.compile()

    # ── Node wrappers with injected dependencies ─────────────────────────────

    def _guardrail(self, state: AgenticRAGState) -> AgenticRAGState:
        return run_guardrail_node(state, config=self._config, ollama=self._ollama)

    def _route_after_guardrail(self, state: AgenticRAGState) -> str:
        return route_after_guardrail(state, config=self._config)

    def _retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        return run_retrieve_node(state, config=self._config, search=self._search)

    def _grade_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        return run_grade_documents_node(state, ollama=self._ollama)

    def _route_after_grading(self, state: AgenticRAGState) -> str:
        return route_after_grading(state, config=self._config)

    def _rewrite_query(self, state: AgenticRAGState) -> AgenticRAGState:
        return run_rewrite_query_node(state, config=self._config, ollama=self._ollama)

    def _generate_answer(self, state: AgenticRAGState) -> AgenticRAGState:
        return run_generate_answer_node(
            state,
            config=self._config,
            context_builder=self._context,
            ollama=self._ollama,
        )

    @staticmethod
    def _to_result(state: AgenticRAGState, *, took_ms: int) -> AgenticRAGResult:
        guardrail = state.get("guardrail_result")
        return AgenticRAGResult(
            question=state["question"],
            answer=state.get("answer") or "No answer generated.",
            sources=[
                source if isinstance(source, SourceItem) else SourceItem.model_validate(source)
                for source in state.get("sources", [])
            ],
            reasoning_steps=[
                step if isinstance(step, ReasoningStep) else ReasoningStep.model_validate(step)
                for step in state.get("reasoning_steps", [])
            ],
            retrieval_attempts=state.get("retrieval_attempts", 0),
            rewritten_query=state.get("rewritten_query"),
            guardrail_score=guardrail.score if guardrail else None,
            search_mode=state.get("search_mode", "hybrid"),
            took_ms=took_ms,
            trace_id=state.get("trace_id"),
        )
