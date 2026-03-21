"""Tests for ContentEmbeddingService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from linkedin.embeddings import ContentEmbeddingService, OPENAI_EMBEDDING_DIM


def make_fake_response(text: str = "test"):
    mock_data = MagicMock()
    mock_data.embedding = np.random.default_rng(42).normal(size=OPENAI_EMBEDDING_DIM).tolist()
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    return mock_response


@pytest.fixture
def service(tmp_path):
    return ContentEmbeddingService(
        cache_dir=str(tmp_path),
        target_dim=128,
        api_key="fake-key-for-testing",
    )


def test_embed_text_shape(service):
    with patch.object(service._client.embeddings, "create", return_value=make_fake_response()):
        result = service.embed_text("Hello LinkedIn")
    assert result.shape == (OPENAI_EMBEDDING_DIM,), (
        f"Expected ({OPENAI_EMBEDDING_DIM},), got {result.shape}"
    )


def test_cache_hit(service):
    with patch.object(
        service._client.embeddings, "create", return_value=make_fake_response()
    ) as mock_api:
        result1 = service.embed_text("test post")
        result2 = service.embed_text("test post")

    assert mock_api.call_count == 1, f"Expected 1 API call, got {mock_api.call_count}"
    np.testing.assert_array_equal(result1, result2)


def test_empty_text_returns_zero(service):
    result = service.embed_text("")
    assert result.shape == (OPENAI_EMBEDDING_DIM,)
    assert np.all(result == 0), "Empty text should return all-zeros"


def test_project_to_model_dim(service):
    embeddings = np.random.default_rng(0).normal(size=(5, OPENAI_EMBEDDING_DIM)).astype(np.float32)
    projected = service.project_to_model_dim(embeddings)
    assert projected.shape == (5, 128), f"Expected (5, 128), got {projected.shape}"


def test_projection_deterministic(service):
    proj1 = service.get_projection_matrix()
    service._projection_matrix = None
    proj2 = service.get_projection_matrix()
    np.testing.assert_array_equal(proj1, proj2)
