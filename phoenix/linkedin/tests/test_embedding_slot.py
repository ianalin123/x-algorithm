# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Test that Phoenix's block_candidate_reduce handles num_item_hashes=3.

This validates the core design choice: the 3rd hash slot can hold a content
embedding without ANY modification to phoenix/recsys_model.py.
"""

import subprocess

import haiku as hk
import jax
import numpy as np
import pytest

from grok import TransformerConfig
from linkedin.config import LinkedInModelConfig
from recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings


def create_test_batch_and_embeddings(
    batch_size: int = 1,
    history_len: int = 4,
    num_candidates: int = 1,
    emb_size: int = 128,
    num_item_hashes: int = 3,
    num_actions: int = 8,
    seed: int = 42,
):
    """Create synthetic batch and embeddings with 3 item hash slots."""
    rng = np.random.default_rng(seed)

    batch = RecsysBatch(
        user_hashes=rng.integers(1, 1000, size=(batch_size, 2)).astype(np.int32),
        history_post_hashes=rng.integers(
            1, 1000, size=(batch_size, history_len, num_item_hashes)
        ).astype(np.int32),
        history_author_hashes=rng.integers(1, 1000, size=(batch_size, history_len, 2)).astype(
            np.int32
        ),
        history_actions=rng.random(size=(batch_size, history_len, num_actions)).astype(np.float32),
        history_product_surface=rng.integers(0, 6, size=(batch_size, history_len)).astype(np.int32),
        candidate_post_hashes=rng.integers(
            1, 1000, size=(batch_size, num_candidates, num_item_hashes)
        ).astype(np.int32),
        candidate_author_hashes=rng.integers(1, 1000, size=(batch_size, num_candidates, 2)).astype(
            np.int32
        ),
        candidate_product_surface=rng.integers(0, 6, size=(batch_size, num_candidates)).astype(
            np.int32
        ),
    )

    # Candidate post embeddings: [B, C, 3, D]
    # Slots 0,1: hash embeddings, Slot 2: content embedding (unit vector)
    candidate_post_embeddings = rng.normal(
        size=(batch_size, num_candidates, num_item_hashes, emb_size)
    ).astype(np.float32)
    content_emb = rng.normal(size=(batch_size, num_candidates, emb_size)).astype(np.float32)
    content_emb = content_emb / (np.linalg.norm(content_emb, axis=-1, keepdims=True) + 1e-8)
    candidate_post_embeddings[:, :, 2, :] = content_emb

    embeddings = RecsysEmbeddings(
        user_embeddings=rng.normal(size=(batch_size, 2, emb_size)).astype(np.float32),
        history_post_embeddings=rng.normal(
            size=(batch_size, history_len, num_item_hashes, emb_size)
        ).astype(np.float32),
        candidate_post_embeddings=candidate_post_embeddings,
        history_author_embeddings=rng.normal(size=(batch_size, history_len, 2, emb_size)).astype(
            np.float32
        ),
        candidate_author_embeddings=rng.normal(
            size=(batch_size, num_candidates, 2, emb_size)
        ).astype(np.float32),
    )

    return batch, embeddings


def test_forward_pass_with_3_hash_slots():
    """PhoenixModel forward pass works with num_item_hashes=3, no NaN, correct shape."""
    config = LinkedInModelConfig()
    phoenix_config = config.make()

    batch, embeddings = create_test_batch_and_embeddings(
        batch_size=1,
        history_len=4,
        num_candidates=1,
        emb_size=config.emb_size,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_actions=config.num_actions,
    )

    def forward(batch, embeddings):
        return phoenix_config.make()(batch, embeddings)

    forward_fn = hk.transform(forward)
    rng_key = jax.random.PRNGKey(42)
    params = forward_fn.init(rng_key, batch, embeddings)
    output = forward_fn.apply(params, rng_key, batch, embeddings)

    logits = np.array(output.logits)

    # Shape: [B=1, C=1, num_actions=8]
    assert logits.shape == (1, 1, 8), f"Expected (1, 1, 8), got {logits.shape}"
    assert not np.any(np.isnan(logits)), "Logits contain NaN values"
    assert np.all(np.isfinite(logits)), "Logits contain infinite values"


def test_output_shape_batch_size_2():
    """Output shape correct for batch_size=2."""
    config = LinkedInModelConfig()
    phoenix_config = config.make()

    batch, embeddings = create_test_batch_and_embeddings(
        batch_size=2,
        history_len=4,
        num_candidates=1,
        emb_size=config.emb_size,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_actions=config.num_actions,
        seed=99,
    )

    def forward(batch, embeddings):
        return phoenix_config.make()(batch, embeddings)

    forward_fn = hk.transform(forward)
    rng_key = jax.random.PRNGKey(7)
    params = forward_fn.init(rng_key, batch, embeddings)
    output = forward_fn.apply(params, rng_key, batch, embeddings)

    assert output.logits.shape == (2, 1, 8), f"Expected (2, 1, 8), got {output.logits.shape}"


def test_phoenix_recsys_model_unmodified():
    """Verify that phoenix/recsys_model.py was NOT modified."""
    result = subprocess.run(
        ["git", "diff", "phoenix/recsys_model.py"],
        capture_output=True,
        text=True,
        cwd="/Users/ianalin/x-algorithm",
    )
    assert result.stdout == "", f"recsys_model.py was modified:\n{result.stdout}"
