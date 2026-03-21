"""LinkedIn-specific batch, embedding, and label dataclasses.

Provides thin wrappers around RecsysBatch and RecsysEmbeddings with LinkedIn-specific
structure, plus label definitions for training targets.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recsys_model import RecsysBatch, RecsysEmbeddings
from linkedin.config import POST_TYPE_MAP


@dataclass
class LinkedInBatch:
    """LinkedIn-specific batch.

    Thin wrapper holding a RecsysBatch with LinkedIn-specific semantics.
    Contains feature data (hashes, actions, product surfaces) but NOT embeddings.
    """

    batch: RecsysBatch


@dataclass
class LinkedInEmbeddings:
    """LinkedIn embeddings wrapper.

    Wraps RecsysEmbeddings and provides factory methods for construction from components.
    Handles the special case where candidate_post_embeddings has 3 hash slots:
    - Slots 0,1: Hash post embeddings
    - Slot 2: Content embeddings (projected to model dimension)
    """

    user_embeddings: np.ndarray
    history_post_embeddings: np.ndarray
    candidate_post_embeddings: np.ndarray
    history_author_embeddings: np.ndarray
    candidate_author_embeddings: np.ndarray

    @property
    def embeddings(self) -> RecsysEmbeddings:
        """Get the underlying RecsysEmbeddings object."""
        return RecsysEmbeddings(
            user_embeddings=self.user_embeddings,
            history_post_embeddings=self.history_post_embeddings,
            candidate_post_embeddings=self.candidate_post_embeddings,
            history_author_embeddings=self.history_author_embeddings,
            candidate_author_embeddings=self.candidate_author_embeddings,
        )

    @classmethod
    def from_components(
        cls,
        user_embeddings: np.ndarray,
        history_post_embeddings: np.ndarray,
        history_author_embeddings: np.ndarray,
        candidate_author_embeddings: np.ndarray,
        content_embeddings: np.ndarray,
        hash_post_embeddings: Optional[np.ndarray] = None,
        emb_size: int = 128,
    ) -> "LinkedInEmbeddings":
        """Create LinkedInEmbeddings from component arrays.

        Args:
            user_embeddings: [B, 2, D] user hash embeddings
            history_post_embeddings: [B, S, 2, D] history post hash embeddings
            history_author_embeddings: [B, S, 2, D] history author hash embeddings
            candidate_author_embeddings: [B, C, 2, D] candidate author hash embeddings
            content_embeddings: [B, C, D] candidate content embeddings (already projected)
            hash_post_embeddings: [B, C, 2, D] candidate post hash embeddings
                If None, initialized as zeros
            emb_size: Embedding size (default 128)

        Returns:
            LinkedInEmbeddings with candidate_post_embeddings [B, C, 3, D]
            where slot 2 contains the content_embeddings
        """
        batch_size = user_embeddings.shape[0]
        candidate_seq_len = content_embeddings.shape[1]

        if hash_post_embeddings is None:
            hash_post_embeddings = np.zeros(
                (batch_size, candidate_seq_len, 2, emb_size), dtype=np.float32
            )

        # Build candidate_post_embeddings [B, C, 3, D]
        # Slots 0,1: hash embeddings, Slot 2: content embeddings
        candidate_post_embeddings = np.zeros(
            (batch_size, candidate_seq_len, 3, emb_size), dtype=np.float32
        )
        candidate_post_embeddings[:, :, :2, :] = hash_post_embeddings
        candidate_post_embeddings[:, :, 2, :] = content_embeddings

        return cls(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embeddings,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            candidate_author_embeddings=candidate_author_embeddings,
        )


@dataclass
class LinkedInLabels:
    """LinkedIn training labels for a batch of candidates.

    Contains engagement targets for training and evaluation.
    """

    reaction_labels: np.ndarray  # [B, C, 6] one-hot or all-zeros (mutually exclusive reactions)
    comment_label: np.ndarray  # [B, C] binary (0 or 1)
    repost_label: np.ndarray  # [B, C] binary (0 or 1)

    def __post_init__(self):
        """Validate label shapes and constraints."""
        assert self.reaction_labels.shape[-1] == 6, (
            f"reaction_labels must have 6 reaction types, got {self.reaction_labels.shape[-1]}"
        )

        # Validate reaction_labels: each row sums to at most 1 (one-hot or all-zeros)
        row_sums = np.sum(self.reaction_labels, axis=-1)
        assert np.all(row_sums <= 1), (
            f"reaction_labels rows must sum to ≤1 (one-hot), found max sum: {np.max(row_sums)}"
        )

        # Validate binary labels
        assert np.all(np.isin(self.comment_label, [0, 1])), "comment_label must be binary (0 or 1)"
        assert np.all(np.isin(self.repost_label, [0, 1])), "repost_label must be binary (0 or 1)"


@dataclass
class LinkedInTrainingBatch:
    """Complete training batch with features, embeddings, and labels."""

    batch: RecsysBatch
    embeddings: LinkedInEmbeddings
    labels: LinkedInLabels


def create_example_linkedin_batch(
    batch_size: int = 1,
    history_len: int = 32,
    emb_size: int = 128,
    seed: int = 42,
) -> LinkedInTrainingBatch:
    """Create synthetic LinkedIn training batch for testing.

    Args:
        batch_size: Number of users in batch (B)
        history_len: Length of user action history (S)
        emb_size: Embedding dimension (D)
        seed: Random seed for reproducibility

    Returns:
        LinkedInTrainingBatch with synthetic data and correct shapes:
        - history_actions: [B, 32, 8]
        - candidate_post_embeddings: [B, 1, 3, 128] (slot 2 has content embeddings)
        - reaction_labels: [B, 1, 6] (one-hot or all-zeros)
        - comment_label, repost_label: [B, 1] binary
    """
    rng = np.random.default_rng(seed)

    # === RecsysBatch features ===
    user_hashes = rng.integers(1, 100000, size=(batch_size, 2), dtype=np.int32)

    history_post_hashes = rng.integers(1, 100000, size=(batch_size, history_len, 3), dtype=np.int32)

    history_author_hashes = rng.integers(
        1, 100000, size=(batch_size, history_len, 2), dtype=np.int32
    )

    # Multi-hot action history: 8 LinkedIn actions, ~30% sparsity
    history_actions = (rng.random(size=(batch_size, history_len, 8)) > 0.7).astype(np.float32)

    history_product_surface = rng.integers(0, 6, size=(batch_size, history_len), dtype=np.int32)

    candidate_seq_len = 1
    candidate_post_hashes = rng.integers(
        1, 100000, size=(batch_size, candidate_seq_len, 3), dtype=np.int32
    )

    candidate_author_hashes = rng.integers(
        1, 100000, size=(batch_size, candidate_seq_len, 2), dtype=np.int32
    )

    candidate_product_surface = rng.integers(
        0, 6, size=(batch_size, candidate_seq_len), dtype=np.int32
    )

    batch = RecsysBatch(
        user_hashes=user_hashes,
        history_post_hashes=history_post_hashes,
        history_author_hashes=history_author_hashes,
        history_actions=history_actions,
        history_product_surface=history_product_surface,
        candidate_post_hashes=candidate_post_hashes,
        candidate_author_hashes=candidate_author_hashes,
        candidate_product_surface=candidate_product_surface,
    )

    # === Embeddings ===
    user_embeddings = rng.normal(size=(batch_size, 2, emb_size), scale=0.01).astype(np.float32)

    history_post_embeddings = rng.normal(
        size=(batch_size, history_len, 3, emb_size), scale=0.01
    ).astype(np.float32)

    history_author_embeddings = rng.normal(
        size=(batch_size, history_len, 2, emb_size), scale=0.01
    ).astype(np.float32)

    candidate_author_embeddings = rng.normal(
        size=(batch_size, candidate_seq_len, 2, emb_size), scale=0.01
    ).astype(np.float32)

    # Content embeddings: random unit vectors
    content_embeddings = rng.normal(size=(batch_size, candidate_seq_len, emb_size)).astype(
        np.float32
    )
    content_embeddings = content_embeddings / (
        np.linalg.norm(content_embeddings, axis=-1, keepdims=True) + 1e-8
    )

    hash_post_embeddings = rng.normal(
        size=(batch_size, candidate_seq_len, 2, emb_size), scale=0.01
    ).astype(np.float32)

    embeddings = LinkedInEmbeddings.from_components(
        user_embeddings=user_embeddings,
        history_post_embeddings=history_post_embeddings,
        history_author_embeddings=history_author_embeddings,
        candidate_author_embeddings=candidate_author_embeddings,
        content_embeddings=content_embeddings,
        hash_post_embeddings=hash_post_embeddings,
        emb_size=emb_size,
    )

    # === Labels ===
    # reaction_labels: [B, C, 6] one-hot or all-zeros
    reaction_labels = np.zeros((batch_size, candidate_seq_len, 6), dtype=np.float32)
    # 50% chance of a reaction, otherwise all-zeros
    for b in range(batch_size):
        for c in range(candidate_seq_len):
            if rng.random() > 0.5:
                reaction_idx = rng.integers(0, 6)
                reaction_labels[b, c, reaction_idx] = 1.0

    comment_label = rng.integers(0, 2, size=(batch_size, candidate_seq_len)).astype(np.float32)
    repost_label = rng.integers(0, 2, size=(batch_size, candidate_seq_len)).astype(np.float32)

    labels = LinkedInLabels(
        reaction_labels=reaction_labels,
        comment_label=comment_label,
        repost_label=repost_label,
    )

    return LinkedInTrainingBatch(
        batch=batch,
        embeddings=embeddings,
        labels=labels,
    )


def encode_post_type(post_type: str) -> int:
    """Encode post type string to product_surface index.

    Args:
        post_type: Post type string (e.g. 'video', 'text', 'image')

    Returns:
        Product surface index from POST_TYPE_MAP, or 0 if not found
    """
    return POST_TYPE_MAP.get(post_type, 0)
