"""The deployed local tokenizer and its complete, thinking-disabled chat template."""

from __future__ import annotations

import hashlib
from pathlib import Path


class LocalChatTokenizer:
    def __init__(self, directory: str | Path):
        from transformers import AutoTokenizer

        self.directory = Path(directory).resolve(strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory,
            local_files_only=True,
            trust_remote_code=False,
        )
        files = sorted(
            p
            for p in self.directory.iterdir()
            if p.is_file()
            and (
                p.name.startswith(
                    ("tokenizer", "special_tokens", "added_tokens", "chat_template")
                )
                or p.name in {"vocab.json", "merges.txt"}
            )
        )
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        self.tokenizer_id = "local-chat-sha256:" + digest.hexdigest()
        if not self.tokenizer.chat_template:
            raise ValueError(
                "the local tokenizer must include the deployed chat template"
            )

    def encode(self, text, *, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens):
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_messages(self, messages):
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(self.encode(self.render(messages), add_special_tokens=False))
        if count != len(tokens):
            raise ValueError("chat-template rendering and prompt tokenization disagree")
        return count
