import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from linkedin.profile_embeddings import ProfileFeatureEmbedder, PROFILE_OUTPUT_DIM


@pytest.fixture
def embedder(tmp_path):
    return ProfileFeatureEmbedder(cache_dir=str(tmp_path), api_key="fake-key")


def test_profile_embedding_shape(embedder):
    with patch.object(embedder, "_get_embedding_service") as mock_svc:
        mock_svc.return_value.embed_text.return_value = np.zeros(1536, dtype=np.float32)
        mock_svc.return_value.project_to_model_dim.return_value = np.zeros(64, dtype=np.float32)
        result = embedder.embed_profile(
            {
                "headline": "VP Engineering",
                "follower_count": 5000,
                "connection_count": 500,
            }
        )
    assert result.shape == (PROFILE_OUTPUT_DIM,), (
        f"Expected ({PROFILE_OUTPUT_DIM},), got {result.shape}"
    )


def test_missing_fields_no_crash(embedder):
    with patch.object(embedder, "_get_embedding_service") as mock_svc:
        mock_svc.return_value.embed_text.return_value = np.zeros(1536, dtype=np.float32)
        mock_svc.return_value.project_to_model_dim.return_value = np.zeros(64, dtype=np.float32)
        result = embedder.embed_profile({"headline": "VP Engineering"})
    assert result.shape == (PROFILE_OUTPUT_DIM,)
    assert not np.any(np.isnan(result)), "Contains NaN"


def test_empty_profile_no_crash(embedder):
    result = embedder.embed_profile({})
    assert result.shape == (PROFILE_OUTPUT_DIM,)
    assert not np.any(np.isnan(result))


def test_batch_shape(embedder):
    with patch.object(embedder, "_get_embedding_service") as mock_svc:
        mock_svc.return_value.embed_text.return_value = np.zeros(1536, dtype=np.float32)
        mock_svc.return_value.project_to_model_dim.return_value = np.zeros(64, dtype=np.float32)
        profiles = [{"headline": f"Profile {i}"} for i in range(5)]
        results = embedder.embed_profiles_batch(profiles)
    assert results.shape == (5, PROFILE_OUTPUT_DIM)


def test_numerical_features_range(embedder):
    result = embedder._encode_numerical(
        {
            "follower_count": 1_000_000,
            "connection_count": 500,
            "location_country": "US",
            "has_education": True,
            "has_work_experience": True,
        }
    )
    assert result.shape == (6,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.5)


def test_projection_matrix_cached(embedder, tmp_path):
    mat1 = embedder._get_projection_matrix()
    mat2 = embedder._get_projection_matrix()
    assert np.array_equal(mat1, mat2)
    assert (tmp_path / "profile_projection_matrix.npy").exists()


def test_projection_matrix_reloaded_from_disk(tmp_path):
    e1 = ProfileFeatureEmbedder(cache_dir=str(tmp_path), api_key="fake-key")
    mat1 = e1._get_projection_matrix()

    e2 = ProfileFeatureEmbedder(cache_dir=str(tmp_path), api_key="fake-key")
    mat2 = e2._get_projection_matrix()
    assert np.array_equal(mat1, mat2)
