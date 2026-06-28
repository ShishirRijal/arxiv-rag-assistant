import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.bot.formatters import format_rag_answer, split_telegram_message
from src.bot.rag_client import RAGClient, RAGClientConfig, RAGClientError


class TelegramFormatterTests(unittest.TestCase):
    def test_format_rag_answer_escapes_html_and_deduplicates_sources(self):
        result = {
            "answer": "Transformer <uses> self-attention.",
            "search_mode": "hybrid",
            "retrieval_attempts": 1,
            "took_ms": 42,
            "sources": [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                },
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                },
            ],
        }

        message = format_rag_answer(result)

        self.assertIn("Transformer &lt;uses&gt; self-attention.", message)
        self.assertIn("<b>Sources</b>", message)
        self.assertEqual(message.count("https://arxiv.org/abs/1706.03762"), 1)
        self.assertIn("<i>mode=hybrid | attempts=1 | took=42ms</i>", message)

    def test_split_telegram_message_splits_long_messages(self):
        text = "paragraph\n\n" * 1000

        chunks = split_telegram_message(text, limit=100)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))


class TelegramRAGClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_agentic_client_posts_to_agentic_endpoint(self):
        mock_response = httpx.Response(
            200,
            json={"answer": "ok"},
            request=httpx.Request("POST", "http://api/api/v1/ask-agentic/"),
        )

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as post:
            client = RAGClient(RAGClientConfig(base_url="http://api", use_agentic_rag=True))

            result = await client.ask("What is attention?")

        self.assertEqual(result, {"answer": "ok"})
        post.assert_awaited_once_with("/api/v1/ask-agentic/", json={"question": "What is attention?"})

    async def test_standard_client_posts_to_standard_endpoint_with_hybrid_flag(self):
        mock_response = httpx.Response(
            200,
            json={"answer": "ok"},
            request=httpx.Request("POST", "http://api/api/v1/ask/"),
        )

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as post:
            client = RAGClient(RAGClientConfig(base_url="http://api", use_agentic_rag=False))

            result = await client.ask("What is attention?")

        self.assertEqual(result, {"answer": "ok"})
        post.assert_awaited_once_with(
            "/api/v1/ask/",
            json={"question": "What is attention?", "use_hybrid": True},
        )

    async def test_client_wraps_http_errors(self):
        mock_response = httpx.Response(
            503,
            json={"detail": "LLM unavailable"},
            request=httpx.Request("POST", "http://api/api/v1/ask-agentic/"),
        )

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = RAGClient(RAGClientConfig(base_url="http://api"))

            with self.assertRaisesRegex(RAGClientError, "LLM unavailable"):
                await client.ask("What is attention?")

    async def test_health_gets_backend_health(self):
        mock_response = httpx.Response(
            200,
            json={"status": "healthy"},
            request=httpx.Request("GET", "http://api/health"),
        )

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)) as get:
            client = RAGClient(RAGClientConfig(base_url="http://api"))

            result = await client.health()

        self.assertEqual(result, {"status": "healthy"})
        get.assert_awaited_once_with("/health")


if __name__ == "__main__":
    unittest.main()
