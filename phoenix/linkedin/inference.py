"""LinkedIn Phoenix inference runner and audience aggregation.

Provides inference scoring for LinkedIn posts and audience-level
engagement prediction with confidence intervals.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from linkedin.config import LINKEDIN_ACTIONS, LinkedInModelConfig
from linkedin.loss import compute_action_probs


@dataclass
class LinkedInPredictions:
    """Per-viewer and aggregate predictions for a LinkedIn post.

    Attributes:
        per_viewer: {viewer_urn: {action_name: probability}}
        aggregate: {action_name: mean_probability}
        expected_counts: {action_name: probability * num_viewers}
    """

    per_viewer: dict
    aggregate: dict
    expected_counts: dict


@dataclass
class AggregateEngagement:
    """Audience-level engagement predictions with confidence intervals.

    Attributes:
        mean_probabilities: {action_name: float}
        expected_engagement: {action_name: int} scaled by follower count
        confidence_interval: {action_name: (lower, upper)}
        engagement_rate: total expected / num_followers
    """

    mean_probabilities: dict
    expected_engagement: dict
    confidence_interval: dict
    engagement_rate: float


class LinkedInInferenceRunner:
    """Scores LinkedIn posts against viewer profiles using a trained Phoenix checkpoint."""

    def __init__(self, checkpoint_dir, config=None):
        self.checkpoint_dir = checkpoint_dir
        self.config = config or LinkedInModelConfig()
        self.params = None
        self.forward_fn = None

    def load(self):
        from linkedin.train import load_checkpoint, make_forward_fn

        self.forward_fn = make_forward_fn(self.config)
        self.params, _, _, _ = load_checkpoint(self.checkpoint_dir)

    def score_post(
        self,
        post_text,
        post_type,
        author_profile,
        viewer_profiles,
        viewer_histories=None,
    ):
        """Score a post against multiple viewer profiles.

        For each viewer, constructs a synthetic batch with dummy embeddings,
        runs the forward pass, and computes per-action probabilities.
        Real content embeddings will be integrated with the full pipeline.
        """
        from linkedin.data import create_example_linkedin_batch
        from linkedin.train import _embeddings_to_jax

        if self.params is None or self.forward_fn is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        action_names = LINKEDIN_ACTIONS
        per_viewer = {}
        rng = jax.random.PRNGKey(0)

        for viewer_urn in viewer_profiles:
            rng, step_rng = jax.random.split(rng)

            example = create_example_linkedin_batch(batch_size=1)
            batch_jax = jax.tree.map(jnp.array, example.batch)
            emb_jax = _embeddings_to_jax(example.embeddings)

            output = self.forward_fn.apply(self.params, step_rng, batch_jax, emb_jax)
            probs = compute_action_probs(output.logits)
            probs_np = np.array(probs[0, 0, :])

            per_viewer[viewer_urn] = {
                action_names[i]: float(probs_np[i]) for i in range(len(action_names))
            }

        num_viewers = len(per_viewer)
        aggregate = {}
        expected_counts = {}
        for action in action_names:
            mean_prob = float(np.mean([per_viewer[v][action] for v in per_viewer]))
            aggregate[action] = mean_prob
            expected_counts[action] = mean_prob * num_viewers

        return LinkedInPredictions(
            per_viewer=per_viewer,
            aggregate=aggregate,
            expected_counts=expected_counts,
        )


def aggregate_predictions(per_viewer_predictions, num_total_followers):
    """Aggregate per-viewer predictions into audience-level engagement estimates.

    Computes mean probabilities, expected counts scaled by follower count,
    and 95% confidence intervals via normal approximation.
    """
    action_names = list(next(iter(per_viewer_predictions.values())).keys())
    n_viewers = len(per_viewer_predictions)

    probs_by_action = {a: [] for a in action_names}
    for viewer_preds in per_viewer_predictions.values():
        for action, prob in viewer_preds.items():
            probs_by_action[action].append(prob)

    mean_probs = {a: float(np.mean(probs)) for a, probs in probs_by_action.items()}
    expected = {a: int(mean_probs[a] * num_total_followers) for a in action_names}

    ci = {}
    for a in action_names:
        std = float(np.std(probs_by_action[a]))
        se = std / np.sqrt(n_viewers) if n_viewers > 1 else 0
        ci[a] = (max(0, mean_probs[a] - 1.96 * se), min(1, mean_probs[a] + 1.96 * se))

    total_expected = sum(expected.values())
    engagement_rate = total_expected / num_total_followers if num_total_followers > 0 else 0

    return AggregateEngagement(
        mean_probabilities=mean_probs,
        expected_engagement=expected,
        confidence_interval=ci,
        engagement_rate=engagement_rate,
    )
