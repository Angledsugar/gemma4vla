"""Tests for FAST action tokenizer."""

import numpy as np
import torch
import pytest

from gemma4vla.model.fast_tokenizer import (
    FASTActionTokenizer,
    ActionTokenSequence,
    FASTActionHead,
)


def test_fallback_encode_decode():
    """Fallback binning tokenizer should encode/decode."""
    tok = FASTActionTokenizer.__new__(FASTActionTokenizer)
    tok.n_bins = 256
    tok._fast = None  # force fallback

    actions = np.random.uniform(-1, 1, (15, 8)).astype(np.float32)
    tokens = tok.encode(actions)
    assert tokens.dtype == np.int64 or tokens.dtype == np.intp
    assert len(tokens) == 15 * 8  # flattened

    recovered = tok.decode(tokens, action_horizon=15, action_dim=8)
    assert recovered.shape == (15, 8)
    assert np.abs(actions - recovered).max() < 0.01  # binning quantization error


def test_action_token_sequence():
    """Build training sequence with prefix + action tokens."""
    tok = FASTActionTokenizer.__new__(FASTActionTokenizer)
    tok.n_bins = 256
    tok._fast = None

    seq_builder = ActionTokenSequence(
        fast_tokenizer=tok,
        vocab_size=262144,
        max_seq_len=512,
    )

    prefix = np.array([1, 100, 200, 300, 400, 2], dtype=np.int64)  # fake prefix tokens
    actions = np.random.uniform(-1, 1, (15, 8)).astype(np.float32)

    result = seq_builder.build_training_sequence(prefix, actions)

    assert result["tokens"].shape == (512,)
    assert result["token_mask"].shape == (512,)
    assert result["ar_mask"].shape == (512,)
    assert result["loss_mask"].shape == (512,)

    # Prefix should be bidirectional (ar_mask=0)
    assert (result["ar_mask"][:6] == 0).all()
    # Action tokens should be causal (ar_mask=1)
    assert result["ar_mask"][6] == 1
    # Loss only on action tokens
    assert not result["loss_mask"][:6].any()
    assert result["loss_mask"][6]


def test_fast_action_head_shape():
    """FASTActionHead should produce (B, S, vocab_size) logits."""
    head = FASTActionHead(hidden_size=2560, vocab_size=262144)
    hidden = torch.randn(2, 100, 2560)
    logits = head(hidden)
    assert logits.shape == (2, 100, 262144)


def test_fast_action_head_loss():
    """Loss should be scalar and only computed on masked positions."""
    head = FASTActionHead(hidden_size=64, vocab_size=1000)
    hidden = torch.randn(2, 50, 64)
    targets = torch.randint(0, 1000, (2, 50))
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[:, 30:] = True  # loss only on last 20 tokens

    loss = head.compute_loss(hidden, targets, mask)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_extract_actions():
    """Extract actions from predicted token sequence."""
    tok = FASTActionTokenizer.__new__(FASTActionTokenizer)
    tok.n_bins = 256
    tok._fast = None

    seq_builder = ActionTokenSequence(
        fast_tokenizer=tok,
        vocab_size=262144,
        max_seq_len=512,
    )

    # Simulate predicted tokens
    prefix = np.array([1, 100, 200], dtype=np.int64)
    actions = np.random.uniform(-1, 1, (15, 8)).astype(np.float32)
    result = seq_builder.build_training_sequence(prefix, actions)

    # Extract
    recovered = seq_builder.extract_actions(
        result["tokens"], prefix_len=3, action_horizon=15, action_dim=8
    )
    assert recovered.shape == (15, 8)
