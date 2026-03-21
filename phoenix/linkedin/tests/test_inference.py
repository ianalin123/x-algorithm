import numpy as np
import pytest

from linkedin.inference import AggregateEngagement, aggregate_predictions


def test_aggregation_valid():
    preds = {f"viewer_{i}": {"like_score": 0.3 + i * 0.01, "comment_score": 0.1} for i in range(10)}
    result = aggregate_predictions(preds, num_total_followers=5000)
    assert isinstance(result, AggregateEngagement)
    assert result.expected_engagement["like_score"] > 0
    assert result.expected_engagement["like_score"] < 5000
    assert result.engagement_rate > 0


def test_aggregation_ci_valid():
    rng = np.random.default_rng(42)
    preds = {f"v_{i}": {"like_score": float(rng.random())} for i in range(50)}
    result = aggregate_predictions(preds, 1000)
    lo, hi = result.confidence_interval["like_score"]
    assert lo <= result.mean_probabilities["like_score"] <= hi


def test_aggregation_scales_with_followers():
    preds = {f"v_{i}": {"like_score": 0.5} for i in range(10)}
    r1 = aggregate_predictions(preds, 1000)
    r2 = aggregate_predictions(preds, 10000)
    assert r2.expected_engagement["like_score"] == 10 * r1.expected_engagement["like_score"]
