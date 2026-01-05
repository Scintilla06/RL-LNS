"""Text tokenizer for MILP.

This module implements the "append variable tokens at the end" strategy.

We append `[VAR_0]...[VAR_{n-1}]` to the end of the MILP text, then run a
decoder-only LLM. Due to causal attention, these appended tokens can attend to
all preceding tokens, so their final-layer hidden states can be used as
variable embeddings.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, List, Tuple

import torch
import torch.nn as nn


VAR_PRED_TOKEN_FORMAT = "[VAR_{}]"


_VAR_REGEX = re.compile(r"x\[(\d+)\]")


def count_variables_in_text(text: str) -> int:
    """Infer number of variables from occurrences like x[0], x[1], ...

    Returns 0 if no variables found. Otherwise returns (max_index + 1).
    """

    matches = _VAR_REGEX.findall(text)
    if not matches:
        return 0
    return max(int(m) for m in matches) + 1


def append_var_tokens(text: str, n_vars: int) -> str:
    """Append `[VAR_i]` tokens at the end of the text."""

    var_tokens = " ".join(VAR_PRED_TOKEN_FORMAT.format(i) for i in range(n_vars))
    return text + "\n" + var_tokens


class TextTokenizerWrapper(nn.Module):
    """Text -> variable hidden states wrapper.

    Interface matches the GNN tokenizer usage in `NeuroSolver`.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 32768,
        chunk_size: int = 8192,  # kept for backward-compatible config; unused
        stride: int = 4096,  # kept for backward-compatible config; unused
        max_vars: int = 1000,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_vars = max_vars

        self._ensure_var_tokens_added()

    def __len__(self) -> int:
        return len(self.tokenizer)

    def _ensure_var_tokens_added(self) -> None:
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = []
        for i in range(self.max_vars):
            token = VAR_PRED_TOKEN_FORMAT.format(i)
            if token not in existing_tokens:
                new_tokens.append(token)

        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    def forward(
        self,
        text: str,
        model: nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        n_vars = count_variables_in_text(text)

        if n_vars == 0:
            hidden_dim = model.config.hidden_size
            return torch.zeros(0, hidden_dim, device=device), 0

        if n_vars > self.max_vars:
            raise ValueError(f"Number of variables ({n_vars}) exceeds max_vars ({self.max_vars})")

        text_with_vars = append_var_tokens(text, n_vars)

        # Important: if truncation is needed, keep the *tail* so appended [VAR_i]
        # tokens remain present in the sequence.
        old_truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        try:
            encoding = self.tokenizer(
                text_with_vars,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        finally:
            self.tokenizer.truncation_side = old_truncation_side

        input_ids = encoding["input_ids"].to(device)
        seq_len = input_ids.size(1)

        # If we had to truncate, the appended tokens stay, but some prefix context
        # may be lost.
        if seq_len >= self.max_length:
            warnings.warn(
                f"Text length reached max_length={self.max_length}; prefix may be truncated.",
                RuntimeWarning,
            )

        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        var_hiddens = hidden[0, -n_vars:, :]
        return var_hiddens, n_vars
