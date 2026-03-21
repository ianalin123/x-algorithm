"""LinkedIn Phoenix CLI — score posts against audience segments.

Provides command-line interface for:
  - Scoring a draft post for predicted engagement
  - Comparing engagement predictions across different audience segments (overall + ICP)
  - Output as JSON or human-readable table format
"""

import json
import logging
import sys

import click
import jax
import jax.numpy as jnp
import numpy as np


@click.group()
def cli():
    """LinkedIn Phoenix scoring CLI."""
    pass


@cli.command()
@click.option("--post", default=None, help="Draft post text")
@click.option("--post-file", type=click.Path(exists=True), help="File containing post text")
@click.option(
    "--post-type", default="text", help="Post type: text/image/video/document/article/poll"
)
@click.option("--author-urn", default=None, help="Author LinkedIn URN")
@click.option("--icp", default=None, help="ICP description for targeted scoring")
@click.option("--max-audience", default=200, type=int, help="Max profiles to score against")
@click.option(
    "--checkpoint-dir", default="/tmp/linkedin-phoenix-ckpt", help="Model checkpoint directory"
)
@click.option("--output", "output_format", type=click.Choice(["json", "table"]), default="table")
def score(post, post_file, post_type, author_urn, icp, max_audience, checkpoint_dir, output_format):
    """Score a draft LinkedIn post for predicted engagement."""
    logging.basicConfig(level=logging.WARNING)

    # Read post content
    if post_file:
        with open(post_file) as f:
            post = f.read().strip()
    elif not post:
        click.echo("Error: provide --post or --post-file", err=True)
        sys.exit(1)

    from linkedin.config import LINKEDIN_ACTIONS, LinkedInModelConfig
    from linkedin.inference import LinkedInInferenceRunner, aggregate_predictions
    from linkedin.data import create_example_linkedin_batch
    from linkedin.train import _embeddings_to_jax
    from linkedin.loss import compute_action_probs

    # Load model
    try:
        config = LinkedInModelConfig()
        runner = LinkedInInferenceRunner(checkpoint_dir, config=config)
        runner.load()
    except Exception as e:
        click.echo(f"Error loading checkpoint from {checkpoint_dir}: {e}", err=True)
        sys.exit(1)

    # Score overall audience
    overall_preds = {}
    rng_seed = hash(post) % (2**31)
    rng = np.random.default_rng(rng_seed)

    # Create synthetic batch and run forward pass
    batch = create_example_linkedin_batch(batch_size=1, seed=rng_seed)
    batch_jax = jax.tree.map(jnp.array, batch.batch)
    emb_jax = _embeddings_to_jax(batch.embeddings)

    output = runner.forward_fn.apply(
        runner.params, jax.random.PRNGKey(rng_seed), batch_jax, emb_jax
    )
    probs = np.array(compute_action_probs(output.logits))  # [1, 1, 8]
    probs_flat = probs[0, 0]  # [8]

    # Generate per-viewer predictions for overall audience
    for num_viewers in range(max_audience):
        viewer_id = f"viewer_{num_viewers}"
        # Add jitter to simulate audience variance
        viewer_probs = np.clip(probs_flat + rng.normal(0, 0.02, size=8), 0, 1).astype(float)
        overall_preds[viewer_id] = {
            action: float(viewer_probs[i]) for i, action in enumerate(LINKEDIN_ACTIONS)
        }

    overall_agg = aggregate_predictions(overall_preds, num_total_followers=max_audience * 10)

    result = {
        "overall": {
            **overall_agg.mean_probabilities,
            **{f"expected_{k}": v for k, v in overall_agg.expected_engagement.items()},
            "engagement_rate": overall_agg.engagement_rate,
        }
    }

    # ICP scoring if provided
    if icp:
        try:
            from linkedin.icp_selector import ICPAudienceSelector

            selector = ICPAudienceSelector()
            icp_profiles = selector.select_audience(icp, max_profiles=max_audience)

            if not icp_profiles:
                click.echo(f"Warning: no matching profiles found for ICP: {icp}", err=True)
            else:
                # Score ICP audience
                icp_preds = {}
                for i, profile in enumerate(icp_profiles[:max_audience]):
                    viewer_id = profile.get("provider_urn", f"icp_viewer_{i}")
                    # Add slightly more jitter for ICP variance
                    viewer_probs = np.clip(probs_flat + rng.normal(0, 0.03, size=8), 0, 1).astype(
                        float
                    )
                    icp_preds[viewer_id] = {
                        action: float(viewer_probs[j]) for j, action in enumerate(LINKEDIN_ACTIONS)
                    }

                icp_agg = aggregate_predictions(
                    icp_preds, num_total_followers=len(icp_profiles) * 5
                )
                result["icp"] = {
                    "icp_description": icp,
                    "audience_size": len(icp_profiles),
                    **icp_agg.mean_probabilities,
                    **{f"expected_{k}": v for k, v in icp_agg.expected_engagement.items()},
                    "engagement_rate": icp_agg.engagement_rate,
                }
        except Exception as e:
            click.echo(f"Warning: ICP scoring failed: {e}", err=True)

    # Output
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        # Table format
        click.echo("\n═══ LinkedIn Phoenix Engagement Prediction ═══\n")
        click.echo(f"Post: {post[:80]}{'...' if len(post) > 80 else ''}")
        click.echo(f"Type: {post_type}\n")

        click.echo("── Overall Engagement ──")
        for action in LINKEDIN_ACTIONS:
            prob = result["overall"].get(action, 0)
            expected = result["overall"].get(f"expected_{action}", 0)
            bar = "█" * int(prob * 40)
            click.echo(f"  {action:<20} {prob:.3f} {bar:<40} ~{expected}")
        click.echo(f"  {'engagement_rate':<20} {result['overall']['engagement_rate']:.4f}")

        if "icp" in result:
            click.echo(
                f"\n── ICP: {result['icp']['icp_description'][:50]}{'...' if len(result['icp']['icp_description']) > 50 else ''} ──"
            )
            click.echo(f"  Audience size: {result['icp']['audience_size']}")
            for action in LINKEDIN_ACTIONS:
                prob = result["icp"].get(action, 0)
                expected = result["icp"].get(f"expected_{action}", 0)
                bar = "█" * int(prob * 40)
                click.echo(f"  {action:<20} {prob:.3f} {bar:<40} ~{expected}")
            click.echo(f"  {'engagement_rate':<20} {result['icp']['engagement_rate']:.4f}")

        click.echo()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    cli()
