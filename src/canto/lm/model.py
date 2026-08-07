"""The acoustic-token language model.

The model reads a sequence of text ids followed by acoustic (codec) token ids
and learns to predict the next token. At inference it is primed with the text
and the codes of a short voice prompt, then sampled autoregressively to produce
new speech in that voice.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import LMConfig
from ..text.vocab import GraphemeVocab
from .embedding import ScaledEmbedding, SinusoidalPositionalEncoding
from .transformer import TransformerBackbone


class AcousticTokenLM(nn.Module):
    """A decoder-only transformer over a joint text + acoustic-token vocabulary."""

    def __init__(self, config: LMConfig, text_vocab: GraphemeVocab | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.text_vocab = text_vocab or GraphemeVocab()

        # Vocabulary layout: text ids, then audio ids, then two audio specials.
        self.text_vocab_size = self.text_vocab.size
        self.audio_offset = self.text_vocab_size
        self.audio_bos_id = self.audio_offset + config.codebook_size
        self.audio_eos_id = self.audio_offset + config.codebook_size + 1
        self.vocab_size = self.audio_eos_id + 1

        self.token_embedding = ScaledEmbedding(self.vocab_size, config.dim)
        self.pos_encoding = SinusoidalPositionalEncoding(config.dim, config.context)
        self.drop = nn.Dropout(config.dropout)
        self.backbone = TransformerBackbone(
            config.dim, config.num_layers, config.num_heads, config.dropout
        )
        self.head = nn.Linear(config.dim, self.vocab_size, bias=False)

    def audio_to_lm(self, codes: torch.Tensor) -> torch.Tensor:
        """Shift codec codebook indices into the LM vocabulary space."""
        return codes + self.audio_offset

    def lm_to_audio(self, ids: torch.Tensor) -> torch.Tensor:
        """Map LM audio ids back to codec codebook indices."""
        return ids - self.audio_offset

    def pack_inputs(
        self,
        text_ids: torch.Tensor,
        prompt_codes: torch.Tensor | None,
        target_codes: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate text, prompt, and target tokens into one LM sequence.

        Shapes: ``text_ids`` is ``(B, L_text)``; ``prompt_codes`` and
        ``target_codes`` are ``(B, L)`` codec indices. Returns ``(B, L_total)``.
        """
        parts = [text_ids]
        batch = text_ids.shape[0]
        device = text_ids.device

        audio_bos = torch.full((batch, 1), self.audio_bos_id, dtype=torch.long, device=device)
        parts.append(audio_bos)
        if prompt_codes is not None:
            parts.append(self.audio_to_lm(prompt_codes))
        if target_codes is not None:
            parts.append(self.audio_to_lm(target_codes))
            audio_eos = torch.full(
                (batch, 1), self.audio_eos_id, dtype=torch.long, device=device
            )
            parts.append(audio_eos)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        text_ids: torch.Tensor,
        prompt_codes: torch.Tensor | None = None,
        target_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward returning next-token logits.

        Returns logits of shape ``(B, L - 1, vocab_size)`` aligned so that
        ``logits[:, t]`` predicts ``seq[:, t + 1]``.
        """
        seq = self.pack_inputs(text_ids, prompt_codes, target_codes)
        inputs = seq[:, :-1]
        x = self.token_embedding(inputs)
        x = self.pos_encoding(x)
        x = self.drop(x)
        x = self.backbone(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        prompt_codes: torch.Tensor | None = None,
        max_new_tokens: int = 200,
        sampler=None,
    ) -> torch.Tensor:
        """Sample acoustic codes autoregressively after the text and prompt.

        ``sampler`` maps a ``(B, vocab)`` logits tensor to ``(B,)`` token ids;
        it defaults to greedy decoding. Returns the newly generated codec codes
        (prompt excluded) as a ``(B, L)`` long tensor. Generation stops for a
        sequence when it emits ``audio_eos``.
        """
        from .sampling import greedy_sample

        if sampler is None:
            sampler = greedy_sample

        was_training = self.training
        self.eval()
        try:
            device = text_ids.device
            ids = self.pack_inputs(text_ids, prompt_codes, None)
            generated: list[torch.Tensor] = []

            for _ in range(max_new_tokens):
                context = ids[:, -self.config.context :]
                x = self.token_embedding(context)
                x = self.pos_encoding(x)
                logits = self.head(self.backbone(x))[:, -1, :]
                # Only acoustic tokens (and the end marker) are valid here.
                allowed = torch.full_like(logits, float("-inf"))
                allowed[:, self.audio_offset : self.audio_offset + self.config.codebook_size] = 0.0
                allowed[:, self.audio_eos_id] = 0.0
                logits = logits + allowed
                next_id = sampler(logits)
                # Stop as soon as any sequence emits the end-of-audio marker.
                if bool((next_id == self.audio_eos_id).any()):
                    break
                generated.append(next_id)
                ids = torch.cat([ids, next_id[:, None]], dim=1)

            if not generated:
                return torch.zeros(ids.shape[0], 0, dtype=torch.long, device=device)
            stacked = torch.stack(generated, dim=1)
            return self.lm_to_audio(stacked)
        finally:
            if was_training:
                self.train()
