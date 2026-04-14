"""FAST action tokenizer for autoregressive action generation.

FAST (Fast Action STructured Tokenizer) converts continuous robot actions
into discrete tokens that can be predicted autoregressively by the language model.

Pipeline:
  Encode: actions (H, D) → DCT → quantize → token sequence
  Decode: token sequence → dequantize → inverse DCT → actions (H, D)

The tokenizer uses the HuggingFace FAST tokenizer from:
  physical-intelligence/fast

This enables π0-FAST style autoregressive action prediction where the
backbone generates action tokens directly (no separate action expert needed).

Reference: π0-FAST paper, OpenPI tokenizer.py
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FASTActionTokenizer:
    """FAST tokenizer for encoding/decoding action chunks as discrete tokens.

    Uses the pretrained FAST tokenizer from HuggingFace which internally
    performs DCT-based compression and quantization.
    """

    def __init__(self, fast_model: str = "physical-intelligence/fast", n_bins: int = 256):
        self.n_bins = n_bins
        self._fast = None

        try:
            from transformers import AutoProcessor
            self._fast = AutoProcessor.from_pretrained(fast_model, trust_remote_code=True)
            logger.info(f"FAST tokenizer loaded from {fast_model}")
        except Exception as e:
            logger.warning(f"Could not load FAST tokenizer: {e}. Using fallback binning.")
            self._fast = None

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Encode continuous actions to discrete tokens.

        Args:
            actions: (H, D) continuous action chunk, values in [-1, 1]
        Returns:
            (num_tokens,) int array of token IDs
        """
        if self._fast is not None:
            tokens = self._fast(actions[None])[0]  # (num_tokens,)
            return np.array(tokens)

        # Fallback: simple per-element binning
        flat = actions.flatten()
        bins = np.linspace(-1, 1, self.n_bins + 1)[:-1]
        return np.digitize(flat, bins) - 1

    def decode(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """Decode discrete tokens back to continuous actions.

        Args:
            tokens: (num_tokens,) int array
            action_horizon: H
            action_dim: D
        Returns:
            (H, D) continuous actions
        """
        if self._fast is not None:
            try:
                actions = self._fast.decode(
                    [tokens.tolist()],
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                )[0]
                return np.array(actions)
            except Exception as e:
                logger.warning(f"FAST decode failed: {e}")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Fallback: reverse binning
        flat = tokens.astype(np.float32)
        flat = flat / self.n_bins * 2 - 1
        total = action_horizon * action_dim
        if len(flat) < total:
            flat = np.pad(flat, (0, total - len(flat)))
        return flat[:total].reshape(action_horizon, action_dim)


class ActionTokenSequence:
    """Build tokenized input/output sequences for FAST autoregressive training.

    Format (following π0-FAST convention):
        [image tokens] [prefix: "Task: ..., State: ...;\n"] [Action: <FAST tokens> |]

    Attention mask:
        prefix: bidirectional (ar_mask=0)
        postfix (action tokens): causal (ar_mask=1)

    Loss mask:
        prefix: no loss
        postfix: loss on action tokens only
    """

    def __init__(
        self,
        fast_tokenizer: FASTActionTokenizer,
        vocab_size: int = 262144,
        max_seq_len: int = 512,
        n_bins: int = 256,
        skip_tokens: int = 128,
    ):
        self.fast = fast_tokenizer
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_bins = n_bins
        self.skip_tokens = skip_tokens

    def _action_tokens_to_vocab(self, action_tokens: np.ndarray) -> np.ndarray:
        """Map FAST token IDs to vocabulary space (high end of vocab)."""
        return self.vocab_size - 1 - self.skip_tokens - action_tokens

    def _vocab_to_action_tokens(self, vocab_tokens: np.ndarray) -> np.ndarray:
        """Map vocabulary tokens back to FAST token IDs."""
        return self.vocab_size - 1 - self.skip_tokens - vocab_tokens

    def build_training_sequence(
        self,
        prefix_token_ids: np.ndarray,  # (L,) tokenized instruction + state
        actions: np.ndarray,            # (H, D) continuous actions
    ) -> dict:
        """Build training sequence with prefix + action tokens.

        Returns:
            tokens: (max_seq_len,) int — full token sequence
            token_mask: (max_seq_len,) bool — True for valid tokens
            ar_mask: (max_seq_len,) int — 0=bidirectional, 1=causal
            loss_mask: (max_seq_len,) bool — True where loss is computed
        """
        # Encode actions to FAST tokens
        action_tokens = self.fast.encode(actions)
        action_vocab = self._action_tokens_to_vocab(action_tokens)

        # Build sequence: [prefix] [action_vocab_tokens]
        prefix_len = len(prefix_token_ids)
        action_len = len(action_vocab)
        total_len = prefix_len + action_len

        tokens = np.zeros(self.max_seq_len, dtype=np.int64)
        token_mask = np.zeros(self.max_seq_len, dtype=bool)
        ar_mask = np.zeros(self.max_seq_len, dtype=np.int64)
        loss_mask = np.zeros(self.max_seq_len, dtype=bool)

        # Fill
        seq_len = min(total_len, self.max_seq_len)
        if prefix_len <= self.max_seq_len:
            tokens[:prefix_len] = prefix_token_ids
            token_mask[:prefix_len] = True
            # prefix: bidirectional
            ar_mask[:prefix_len] = 0
            loss_mask[:prefix_len] = False

        action_start = prefix_len
        action_end = min(action_start + action_len, self.max_seq_len)
        if action_start < self.max_seq_len:
            n_action = action_end - action_start
            tokens[action_start:action_end] = action_vocab[:n_action]
            token_mask[action_start:action_end] = True
            # action tokens: causal
            ar_mask[action_start:action_end] = 1
            loss_mask[action_start:action_end] = True

        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "ar_mask": ar_mask,
            "loss_mask": loss_mask,
            "prefix_len": prefix_len,
            "action_len": min(action_len, self.max_seq_len - prefix_len),
        }

    def extract_actions(
        self,
        predicted_tokens: np.ndarray,  # (seq_len,) predicted token IDs
        prefix_len: int,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        """Extract and decode actions from predicted token sequence.

        Returns: (H, D) continuous actions
        """
        # Get action portion of predicted tokens
        action_vocab_tokens = predicted_tokens[prefix_len:]
        # Remove padding
        action_vocab_tokens = action_vocab_tokens[action_vocab_tokens != 0]

        if len(action_vocab_tokens) == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Map back to FAST token space
        action_tokens = self._vocab_to_action_tokens(action_vocab_tokens)
        return self.fast.decode(action_tokens, action_horizon, action_dim)


class FASTActionHead(torch.nn.Module):
    """Autoregressive action head for FAST token prediction.

    Takes backbone hidden states and generates action tokens
    autoregressively using the backbone's own vocabulary.

    This is an alternative to the flow matching action expert.
    Instead of a separate transformer, action tokens are predicted
    by the backbone itself (like next-token prediction in LLMs).
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        vocab_size: int = 262144,
        action_horizon: int = 15,
        action_dim: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # LM head for action token prediction (shares backbone embedding space)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict logits over vocabulary for each position.

        Args:
            hidden_states: (B, S, hidden_size)
        Returns:
            (B, S, vocab_size) logits
        """
        return self.lm_head(hidden_states)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,  # (B, S, hidden_size)
        target_tokens: torch.Tensor,   # (B, S) target token IDs
        loss_mask: torch.Tensor,       # (B, S) bool — where to compute loss
    ) -> torch.Tensor:
        """Cross-entropy loss on action tokens only.

        Loss is computed only at positions where loss_mask=True
        (action token positions, not prefix positions).
        """
        logits = self.forward(hidden_states)  # (B, S, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = target_tokens[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()

        # Compute loss only on masked positions
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_mask.shape)
        masked_loss = (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp(min=1)
        return masked_loss
