"""Versioned full-code embedding shared by ingestion and trajectory queries."""

import hashlib
from pathlib import Path

VERSION = "qwen3-full-code-v1"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MODEL_PATH = (
    Path.home()
    / f"hf-cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/{REVISION}"
)


def code_text(code):
    if not isinstance(code, str) or not code.strip():
        raise ValueError("full-code embedding requires nonempty code")
    return code


class FullCodeEmbedder:
    """No character or tokenizer truncation. Oversize inputs fail explicitly.

    Local inference avoids changing shared embedding-service settings. One record
    at a time bounds memory. Loading is lazy so imports never allocate a GPU.
    """

    def __init__(self, device="cuda:0", max_tokens=32768):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_tokens = max_tokens
        torch.set_num_threads(4)
        if device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(0.06, torch.device(device))
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        self.model = (
            AutoModel.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.last_receipt = None

    def embed(self, code):
        text = code_text(code)
        inputs = self.tokenizer(text, truncation=False, return_tensors="pt")
        count = inputs.input_ids.shape[1]
        if count > self.max_tokens:
            raise ValueError(
                f"full code has {count} tokens, limit {self.max_tokens}; nothing was truncated"
            )
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            vector = (
                self.torch.nn.functional.normalize(hidden[:, -1].float(), dim=1)[0]
                .cpu()
                .numpy()
            )
        self.last_receipt = {
            "version": VERSION,
            "chars": len(text),
            "tokens": count,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "truncated": False,
        }
        return vector
