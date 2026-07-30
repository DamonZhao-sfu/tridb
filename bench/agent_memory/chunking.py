"""Shared chunking for the agent-memory benchmarks.

``TiktokenSentenceChunker`` moved here VERBATIM from
``longmemeval_pipeline.py`` so the deterministic GEM ingest strategy and the
existing embedRAG pipeline share one chunker. That sharing is the point: the
GEM regression gate (docs/agent_memory_gem_implementation_plan_v0.1.0.md §10,
G2) asserts that routing the same corpus through the new interface does not
move the retrieval numbers, and it can only assert that honestly if both sides
chunk identically. A second, drifting copy would quietly invalidate the
comparison.

No behaviour change: this is a pure move.
"""

from __future__ import annotations


class TiktokenSentenceChunker:
    """MemoryAgentBench-compatible 4,096-token sentence packing."""

    _JOIN_MARGIN_TOKENS = 32

    def __init__(
        self,
        *,
        chunk_size: int = 4096,
        tokenizer_model: str = "gpt-4o-mini",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken is required; install requirements-agent-memory.txt"
            ) from exc
        try:
            import nltk
        except ImportError as exc:
            raise RuntimeError(
                "nltk is required; install requirements-agent-memory.txt"
            ) from exc
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.nltk = nltk
        self.chunk_size = chunk_size
        self.tokenizer_model = tokenizer_model

    def _sentences(self, text: str) -> list[str]:
        try:
            return self.nltk.sent_tokenize(text)
        except LookupError as exc:
            raise RuntimeError(
                "NLTK punkt data is missing; run: "
                "python -m nltk.downloader punkt punkt_tab"
            ) from exc

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0
        for sentence in self._sentences(text):
            tokens = self.encoding.encode(
                sentence,
                allowed_special={"<|endoftext|>"},
            )
            if len(tokens) > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                for start in range(0, len(tokens), self.chunk_size):
                    chunks.append(
                        self.encoding.decode(tokens[start : start + self.chunk_size])
                    )
                continue
            packing_limit = max(1, self.chunk_size - self._JOIN_MARGIN_TOKENS)
            if current_sentences and current_tokens + len(tokens) > packing_limit:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            current_sentences.append(sentence)
            current_tokens += len(tokens)
        if current_sentences:
            chunks.append(" ".join(current_sentences))
        bounded: list[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            tokens = self.encoding.encode(
                chunk,
                allowed_special={"<|endoftext|>"},
            )
            for start in range(0, len(tokens), self.chunk_size):
                bounded.append(
                    self.encoding.decode(tokens[start : start + self.chunk_size])
                )
        return bounded

    def count(self, text: str) -> int:
        return len(
            self.encoding.encode(
                text,
                allowed_special={"<|endoftext|>"},
            )
        )
