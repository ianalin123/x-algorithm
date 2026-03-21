import importlib
from typing import Any, NamedTuple

jax = importlib.import_module("jax")
jnp = importlib.import_module("jax.numpy")
optax = importlib.import_module("optax")

from linkedin.config import (
    LINKEDIN_ACTIONS,
    LINKEDIN_INDEPENDENT_ACTIONS,
    LINKEDIN_REACTION_ACTIONS,
)


class LinkedInLabels(NamedTuple):
    reaction_labels: Any
    comment_label: Any
    repost_label: Any


def linkedin_hybrid_loss(
    logits: Any,
    labels: LinkedInLabels,
) -> Any:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits rank 3 [B, C, 8], got shape {logits.shape}")

    expected_num_actions = len(LINKEDIN_ACTIONS)
    expected_num_reactions = len(LINKEDIN_REACTION_ACTIONS)
    expected_num_independent = len(LINKEDIN_INDEPENDENT_ACTIONS)

    if logits.shape[-1] != expected_num_actions:
        raise ValueError(f"Expected logits last dim {expected_num_actions}, got {logits.shape[-1]}")

    reaction_logits = logits[:, :, :expected_num_reactions]
    independent_logits = logits[:, :, expected_num_reactions:]

    if independent_logits.shape[-1] != expected_num_independent:
        raise ValueError(
            "Independent logits width mismatch: "
            f"expected {expected_num_independent}, got {independent_logits.shape[-1]}"
        )

    comment_logit = independent_logits[:, :, 0]
    repost_logit = independent_logits[:, :, 1]

    reaction_labels = labels.reaction_labels.astype(jnp.float32)
    comment_label = labels.comment_label.astype(jnp.float32)
    repost_label = labels.repost_label.astype(jnp.float32)

    has_reaction = jnp.any(reaction_labels > 0, axis=-1).astype(jnp.float32)
    reaction_ce = optax.softmax_cross_entropy(
        logits=reaction_logits.astype(jnp.float32),
        labels=reaction_labels,
    )
    reaction_loss = jnp.mean(reaction_ce * has_reaction)

    comment_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            logits=comment_logit.astype(jnp.float32),
            labels=comment_label,
        )
    )

    repost_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            logits=repost_logit.astype(jnp.float32),
            labels=repost_label,
        )
    )

    total_loss = reaction_loss + comment_loss + repost_loss
    return total_loss


def compute_action_probs(logits: Any) -> Any:
    expected_num_reactions = len(LINKEDIN_REACTION_ACTIONS)
    reaction_probs = jax.nn.softmax(logits[:, :, :expected_num_reactions], axis=-1)
    independent_probs = jax.nn.sigmoid(logits[:, :, expected_num_reactions:])
    return jnp.concatenate([reaction_probs, independent_probs], axis=-1)
