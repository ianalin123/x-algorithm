import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linkedin.config import LinkedInModelConfig
from linkedin.data import create_example_linkedin_batch
from linkedin.train import (
    _make_training_step,
    create_optimizer,
    init_training,
    load_checkpoint,
    make_forward_fn,
    save_checkpoint,
)


def test_optimizer_creates():
    opt = create_optimizer(learning_rate=1e-4, warmup_steps=10, total_steps=100)
    assert opt is not None


def _to_jax_embeddings(li_emb):
    from recsys_model import RecsysEmbeddings

    emb = li_emb.embeddings
    return RecsysEmbeddings(
        user_embeddings=jnp.array(emb.user_embeddings),
        history_post_embeddings=jnp.array(emb.history_post_embeddings),
        candidate_post_embeddings=jnp.array(emb.candidate_post_embeddings),
        history_author_embeddings=jnp.array(emb.history_author_embeddings),
        candidate_author_embeddings=jnp.array(emb.candidate_author_embeddings),
    )


def test_forward_fn_produces_output():
    config = LinkedInModelConfig()
    forward_fn = make_forward_fn(config)
    batch = create_example_linkedin_batch(batch_size=1)
    rng = jax.random.PRNGKey(42)
    batch_jax = jax.tree.map(jnp.array, batch.batch)
    emb_jax = _to_jax_embeddings(batch.embeddings)
    params = forward_fn.init(rng, batch_jax, emb_jax)
    output = forward_fn.apply(params, rng, batch_jax, emb_jax)
    assert output.logits.shape == (1, 1, 8)
    assert not jnp.any(jnp.isnan(output.logits))


def test_training_step_runs():
    config = LinkedInModelConfig()
    forward_fn = make_forward_fn(config)
    optimizer = create_optimizer(learning_rate=1e-3, warmup_steps=1, total_steps=10)
    params, opt_state, rng = init_training(config, forward_fn, optimizer)
    step_fn = _make_training_step(forward_fn, optimizer)
    batch = create_example_linkedin_batch(batch_size=2, seed=42)
    from linkedin.train import _batch_to_jax, _call_step

    batch_jax, emb_jax, labels_jax = _batch_to_jax(batch)
    rng, step_rng = jax.random.split(rng)
    new_params, new_opt_state, loss, logits = _call_step(
        step_fn, params, opt_state, step_rng, batch_jax, emb_jax, labels_jax
    )
    assert jnp.isfinite(loss)
    assert logits.shape == (2, 1, 8)


def test_checkpoint_roundtrip():
    config = LinkedInModelConfig()
    forward_fn = make_forward_fn(config)
    optimizer = create_optimizer()
    params, _, rng = init_training(config, forward_fn, optimizer)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(params, None, step=42, metrics={"loss": 0.5}, checkpoint_dir=tmpdir)
        loaded_params, _, loaded_step, loaded_metrics = load_checkpoint(tmpdir, config, optimizer)
        assert loaded_step == 42
        assert loaded_metrics["loss"] == 0.5
        orig_leaves = jax.tree.leaves(params)
        loaded_leaves = jax.tree.leaves(loaded_params)
        assert len(orig_leaves) == len(loaded_leaves)
        for o, l in zip(orig_leaves, loaded_leaves):
            np.testing.assert_allclose(np.array(o), np.array(l), atol=1e-6)
