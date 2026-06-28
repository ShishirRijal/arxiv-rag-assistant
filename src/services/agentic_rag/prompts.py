"""Prompt templates for the agentic RAG workflow."""

GUARDRAIL_PROMPT = """You are a scope validator for an arXiv research paper assistant.

The assistant only answers questions grounded in indexed arXiv papers about:
- computer science
- artificial intelligence
- machine learning
- natural language processing
- neural networks
- retrieval, search, and RAG systems
- model architecture, training, evaluation, and interpretability

User question:
{question}

Score whether this question belongs to that research-paper domain.

Scoring guide:
- 80-100: clearly about CS/AI/ML research or papers
- 60-79: probably related to CS/AI/ML research, but broad or underspecified
- 40-59: ambiguous or only weakly related
- 0-39: outside the indexed research-paper domain

Respond with JSON only:
{{
  "score": 0,
  "reason": "short explanation"
}}
"""

GRADE_DOCUMENTS_PROMPT = """You are grading whether retrieved arXiv paper chunks are useful for answering a user question.

User question:
{question}

Retrieved context:
{context}

Decide whether the context contains enough relevant evidence to answer the question.

Respond with JSON only:
{{
  "is_relevant": true,
  "score": 0.0,
  "reason": "short explanation"
}}
"""

REWRITE_QUERY_PROMPT = """You rewrite user questions into better search queries for arXiv paper retrieval.

Original user question:
{question}

Current failed or weak search query:
{current_query}

Rewrite the query to improve retrieval from CS/AI/ML research papers.
Keep the user's intent. Add useful technical terms only when appropriate.

Respond with JSON only:
{{
  "rewritten_query": "improved search query",
  "reason": "short explanation"
}}
"""
