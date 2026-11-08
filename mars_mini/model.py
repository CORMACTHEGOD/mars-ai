"""An original tiny autoregressive Transformer language model.

All parameters start random. No pretrained weights, remote model API or
downloaded training corpus are used by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MarsConfig:
    vocab_size: int = 256  # All possible UTF-8 byte values
    max_seq_len: int = 96
    embedding_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        # Legacy Mars-1/Mars-2 still use vocab_size=256. The new BPE model
        # adds learned merges after the initial 256 byte symbols, without
        # changing any existing checkpoint or browser model.
        if not 256 <= self.vocab_size <= 4096:
            raise ValueError("Mars vocabulary size must be 256..4096")
        if self.embedding_dim <= 0 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if self.max_seq_len < 8 or self.num_heads < 1 or self.num_layers < 1:
            raise ValueError("Model configuration is too small")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CausalAttention(nn.Module):
    def __init__(self, cfg: MarsConfig):
        super().__init__()
        self.heads = cfg.num_heads
        self.head_dim = cfg.embedding_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embedding_dim, 3 * cfg.embedding_dim)
        self.projection = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(
            ~self.causal_mask[:length, :length], torch.finfo(scores.dtype).min
        )
        probabilities = self.dropout(F.softmax(scores, dim=-1))
        attended = probabilities @ v
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, channels)
        return self.dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MarsConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.embedding_dim)
        self.attention = CausalAttention(cfg)
        self.feed_forward_norm = nn.LayerNorm(cfg.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class MarsMini(nn.Module):
    def __init__(self, cfg: MarsConfig | None = None):
        super().__init__()
        self.cfg = cfg or MarsConfig()
        self.cfg.validate()
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embedding_dim)
        self.position_embedding = nn.Embedding(self.cfg.max_seq_len, self.cfg.embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(self.cfg) for _ in range(self.cfg.num_layers)])
        self.final_norm = nn.LayerNorm(self.cfg.embedding_dim)
        self.language_head = nn.Linear(self.cfg.embedding_dim, self.cfg.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight  # Tied embeddings
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("Expected [batch, sequence] token tensor")
        if tokens.size(1) > self.cfg.max_seq_len:
            raise ValueError("Sequence exceeds model context limit")
        if tokens.dtype != torch.long:
            raise ValueError("Token indices must be torch.long")

        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        x = self.final_norm(self.blocks(x))
        logits = self.language_head(x)
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("Targets must match input tokens")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.85,
        top_k: int = 40,
    ) -> torch.Tensor:
        if temperature <= 0 or max_new_tokens < 0 or top_k < 1:
            raise ValueError("Invalid generation settings")
        self.eval()
        tokens = prefix
        for _ in range(max_new_tokens):
            context = tokens[:, -self.cfg.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k < self.cfg.vocab_size:
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits = next_logits.masked_fill(
                    next_logits < values[:, [-1]], -float("inf")
                )
            probability = F.softmax(next_logits, dim=-1)
            sampled = torch.multinomial(probability, num_samples=1)
            tokens = torch.cat((tokens, sampled), dim=1)
        return tokens

    def metadata(self) -> dict:
        return {"architecture": "Mars-1 Mini decoder-only Transformer", "config": asdict(self.cfg)}


def encode(text: str) -> list[int]:
    """Byte tokenizer: our whole vocabulary is UTF-8 bytes, no model downloads."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")
