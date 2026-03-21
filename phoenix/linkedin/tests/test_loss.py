import importlib

import numpy as np

from linkedin.loss import LinkedInLabels, compute_action_probs, linkedin_hybrid_loss

jax = importlib.import_module("jax")
jnp = importlib.import_module("jax.numpy")


def make_labels(B=1, C=1, reaction_idx=None, comment=0.0, repost=0.0):
    reaction = np.zeros((B, C, 6), dtype=np.float32)
    if reaction_idx is not None:
        reaction[:, :, reaction_idx] = 1.0
    return LinkedInLabels(
        reaction_labels=jnp.array(reaction),
        comment_label=jnp.array(np.full((B, C), comment, dtype=np.float32)),
        repost_label=jnp.array(np.full((B, C), repost, dtype=np.float32)),
    )


def test_loss_positive_finite():
    logits = jnp.zeros((1, 1, 8))
    labels = make_labels(reaction_idx=0, comment=1.0, repost=0.0)
    loss = linkedin_hybrid_loss(logits, labels)
    assert jnp.isfinite(loss), f"Loss not finite: {loss}"
    assert loss > 0


def test_loss_negative_finite():
    logits = jnp.zeros((1, 1, 8))
    labels = make_labels(reaction_idx=None, comment=0.0, repost=0.0)
    loss = linkedin_hybrid_loss(logits, labels)
    assert jnp.isfinite(loss), f"Loss not finite for negative: {loss}"


def test_loss_no_nan():
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (4, 1, 8))
    labels = make_labels(B=4, reaction_idx=2, comment=1.0, repost=0.0)
    loss = linkedin_hybrid_loss(logits, labels)
    assert not jnp.isnan(loss), "Loss is NaN"


def test_probs_sum_reaction():
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 8))
    probs = compute_action_probs(logits)
    reaction_sum = probs[:, :, :6].sum(axis=-1)
    np.testing.assert_allclose(np.array(reaction_sum), 1.0, atol=1e-5)


def test_probs_independent_range():
    logits = jax.random.normal(jax.random.PRNGKey(1), (2, 1, 8))
    probs = compute_action_probs(logits)
    independent = np.array(probs[:, :, 6:])
    assert np.all(independent >= 0) and np.all(independent <= 1)
