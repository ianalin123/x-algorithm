"""LinkedIn Phoenix training pipeline.

Training step, optimizer, runner with wandb, checkpointing, and evaluation.
"""

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


def create_optimizer(
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


def make_forward_fn(config):
    phoenix_config = config.make()

    def forward(batch, embeddings):
        return phoenix_config.make()(batch, embeddings)

    return hk.transform(forward)


def _embeddings_to_jax(li_emb):
    from recsys_model import RecsysEmbeddings

    emb = li_emb.embeddings if hasattr(li_emb, "embeddings") else li_emb
    return RecsysEmbeddings(
        user_embeddings=jnp.array(emb.user_embeddings),
        history_post_embeddings=jnp.array(emb.history_post_embeddings),
        candidate_post_embeddings=jnp.array(emb.candidate_post_embeddings),
        history_author_embeddings=jnp.array(emb.history_author_embeddings),
        candidate_author_embeddings=jnp.array(emb.candidate_author_embeddings),
    )


def _batch_to_jax(training_batch):
    from linkedin.loss import LinkedInLabels

    batch_jax = jax.tree.map(jnp.array, training_batch.batch)
    emb_jax = _embeddings_to_jax(training_batch.embeddings)
    labels_jax = LinkedInLabels(
        reaction_labels=jnp.array(training_batch.labels.reaction_labels),
        comment_label=jnp.array(training_batch.labels.comment_label),
        repost_label=jnp.array(training_batch.labels.repost_label),
    )
    return batch_jax, emb_jax, labels_jax


def _make_training_step(forward_fn, optimizer):
    from linkedin.loss import linkedin_hybrid_loss, LinkedInLabels
    from recsys_model import RecsysEmbeddings

    @jax.jit
    def step(
        params,
        opt_state,
        rng,
        batch_jax,
        user_emb,
        hist_post_emb,
        cand_post_emb,
        hist_author_emb,
        cand_author_emb,
        reaction_labels,
        comment_label,
        repost_label,
    ):
        def loss_fn(params):
            emb = RecsysEmbeddings(
                user_embeddings=user_emb,
                history_post_embeddings=hist_post_emb,
                candidate_post_embeddings=cand_post_emb,
                history_author_embeddings=hist_author_emb,
                candidate_author_embeddings=cand_author_emb,
            )
            output = forward_fn.apply(params, rng, batch_jax, emb)
            labels = LinkedInLabels(
                reaction_labels=reaction_labels,
                comment_label=comment_label,
                repost_label=repost_label,
            )
            loss = linkedin_hybrid_loss(output.logits, labels)
            return loss, (loss, output.logits)

        grads, (loss, logits) = jax.grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, logits

    return step


def _call_step(step_fn, params, opt_state, rng, batch_jax, emb_jax, labels_jax):
    return step_fn(
        params,
        opt_state,
        rng,
        batch_jax,
        emb_jax.user_embeddings,
        emb_jax.history_post_embeddings,
        emb_jax.candidate_post_embeddings,
        emb_jax.history_author_embeddings,
        emb_jax.candidate_author_embeddings,
        labels_jax.reaction_labels,
        labels_jax.comment_label,
        labels_jax.repost_label,
    )


def init_training(config, forward_fn, optimizer):
    from linkedin.data import create_example_linkedin_batch

    rng = jax.random.PRNGKey(42)
    batch = create_example_linkedin_batch(batch_size=1)
    batch_jax = jax.tree.map(jnp.array, batch.batch)
    emb_jax = _embeddings_to_jax(batch.embeddings)
    params = forward_fn.init(rng, batch_jax, emb_jax)
    opt_state = optimizer.init(params)
    return params, opt_state, rng


def save_checkpoint(params, opt_state, step, metrics, checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    params_np = jax.tree.map(np.array, params)
    leaves = jax.tree.leaves(params_np)
    structure = jax.tree.structure(params)

    np.savez(str(ckpt_dir / "params.npz"), *leaves)
    with open(str(ckpt_dir / "param_structure.pkl"), "wb") as f:
        pickle.dump(structure, f)
    with open(str(ckpt_dir / "metadata.pkl"), "wb") as f:
        pickle.dump({"step": step, "metrics": metrics}, f)

    logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def load_checkpoint(checkpoint_dir, config=None, optimizer=None):
    ckpt_dir = Path(checkpoint_dir)

    with open(str(ckpt_dir / "param_structure.pkl"), "rb") as f:
        structure = pickle.load(f)

    data = np.load(str(ckpt_dir / "params.npz"))
    leaves = [jnp.array(data[f"arr_{i}"]) for i in range(len(data.files))]
    params = jax.tree.unflatten(structure, leaves)

    with open(str(ckpt_dir / "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    opt_state = optimizer.init(params) if optimizer else None
    return params, opt_state, meta.get("step", 0), meta.get("metrics", {})


def compute_auc_roc(predictions, labels, action_names):
    from sklearn.metrics import roc_auc_score

    results = {}
    for i, name in enumerate(action_names):
        pred_col = predictions[:, i] if predictions.ndim > 1 else predictions
        label_col = labels[:, i] if labels.ndim > 1 else labels
        if len(np.unique(label_col)) < 2:
            results[name] = float("nan")
            continue
        results[name] = roc_auc_score(label_col, pred_col)
    return results


def run_overfit_test(steps: int = 100):
    from linkedin.config import LinkedInModelConfig
    from linkedin.data import create_example_linkedin_batch

    config = LinkedInModelConfig()
    forward_fn = make_forward_fn(config)
    optimizer = create_optimizer(learning_rate=1e-3, warmup_steps=10, total_steps=steps)
    params, opt_state, rng = init_training(config, forward_fn, optimizer)
    step_fn = _make_training_step(forward_fn, optimizer)

    batch = create_example_linkedin_batch(batch_size=4, seed=42)
    batch_jax, emb_jax, labels_jax = _batch_to_jax(batch)

    initial_loss = None
    final_loss = None

    for i in range(steps):
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss, _ = _call_step(
            step_fn, params, opt_state, step_rng, batch_jax, emb_jax, labels_jax
        )
        loss_val = float(loss)
        if i == 0:
            initial_loss = loss_val
        if i == steps - 1:
            final_loss = loss_val
        if i % 20 == 0:
            print(f"step {i}: loss={loss_val:.4f}")

    print(f"step {steps - 1}: loss={final_loss:.4f}")
    print(f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.8, (
        f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
    )
    print("OVERFIT TEST PASSED")
    return initial_loss, final_loss


def train(
    config=None,
    checkpoint_dir: str = "/tmp/linkedin-phoenix-ckpt",
    max_steps: int = 10000,
    max_epochs: int = 20,
    eval_every_n_steps: int = 100,
    early_stopping_patience: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    data_subset: Optional[int] = None,
    wandb_project: Optional[str] = None,
    dsn: Optional[str] = None,
):
    from linkedin.config import LinkedInModelConfig
    from linkedin.data import create_example_linkedin_batch

    if config is None:
        config = LinkedInModelConfig()

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            config={
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "emb_size": config.emb_size,
                "history_seq_len": config.history_seq_len,
            },
        )

    forward_fn = make_forward_fn(config)
    optimizer = create_optimizer(
        learning_rate=learning_rate,
        warmup_steps=min(500, max_steps // 10),
        total_steps=max_steps,
    )
    params, opt_state, rng = init_training(config, forward_fn, optimizer)
    step_fn = _make_training_step(forward_fn, optimizer)

    param_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Model parameters: {param_count:,}")

    has_db = dsn or os.environ.get("LINKEDIN_DB_DSN")
    assembler = None
    histories = None
    training_pairs = None
    posts = None

    if has_db:
        logger.info("Loading real data from DB")
        from linkedin.data_pipeline import (
            LinkedInBatchAssembler,
            LinkedInDataLoader,
            NegativeSampler,
            UserHistoryBuilder,
        )

        loader = LinkedInDataLoader(dsn=dsn)
        reactions = loader.load_reactions(limit=data_subset)
        posts = loader.load_posts(limit=data_subset)
        builder = UserHistoryBuilder(max_history_len=config.history_seq_len)
        histories = builder.build_histories(reactions, posts)
        sampler = NegativeSampler(negative_ratio=5)
        training_pairs = sampler.build_training_pairs(reactions, posts)
        assembler = LinkedInBatchAssembler(config)
        logger.info(f"Training pairs: {len(training_pairs)}, Users: {len(histories)}")
    else:
        logger.info("No DB — using synthetic data")

    best_loss = float("inf")
    patience_counter = 0

    for step_num in range(max_steps):
        rng, step_rng = jax.random.split(rng)

        if assembler is not None and training_pairs is not None:
            batch_indices = np.random.randint(0, len(training_pairs), size=batch_size).tolist()
            batch = assembler.assemble_batch(training_pairs, histories, posts, batch_indices)
        else:
            batch = create_example_linkedin_batch(batch_size=batch_size, seed=step_num)

        batch_jax, emb_jax, labels_jax = _batch_to_jax(batch)
        params, opt_state, loss, logits = _call_step(
            step_fn, params, opt_state, step_rng, batch_jax, emb_jax, labels_jax
        )
        loss_val = float(loss)

        if step_num % 10 == 0:
            print(f"step {step_num}: loss={loss_val:.4f}")
            if use_wandb:
                import wandb

                wandb.log({"train/loss": loss_val, "train/step": step_num})

        if step_num % eval_every_n_steps == 0 and step_num > 0:
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
                save_checkpoint(
                    params,
                    opt_state,
                    step_num,
                    {"best_loss": best_loss},
                    checkpoint_dir,
                )
                logger.info(f"New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1

            if use_wandb:
                import wandb

                wandb.log({"val/best_loss": best_loss, "val/patience": patience_counter})

            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping at step {step_num} (patience={early_stopping_patience})"
                )
                break

    save_checkpoint(
        params,
        opt_state,
        step_num,
        {"final_loss": loss_val, "best_loss": best_loss},
        checkpoint_dir,
    )
    print(
        f"Training complete. Steps: {step_num + 1}, "
        f"Final loss: {loss_val:.4f}, Best loss: {best_loss:.4f}"
    )

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LinkedIn Phoenix model")
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--data-subset", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/linkedin-phoenix-ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--overfit-test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.overfit_test:
        run_overfit_test(steps=100)
    else:
        train(
            checkpoint_dir=args.checkpoint_dir,
            max_steps=args.max_steps,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_subset=args.data_subset,
            wandb_project=args.wandb_project,
            dsn=args.dsn,
        )
