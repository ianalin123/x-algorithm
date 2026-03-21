import numpy as np
import pytest
from linkedin.hashing import (
    hash_user,
    hash_post,
    hash_author,
    USER_TABLE_SIZE,
    POST_TABLE_SIZE,
    AUTHOR_TABLE_SIZE,
)
from linkedin.config import LinkedInModelConfig


@pytest.fixture
def config():
    return LinkedInModelConfig()


def test_hash_deterministic(config):
    h1 = hash_user("urn:li:person:abc123", config.hash_config)
    h2 = hash_user("urn:li:person:abc123", config.hash_config)
    np.testing.assert_array_equal(h1, h2)


def test_hash_distinct(config):
    h1 = hash_user("urn:li:person:abc123", config.hash_config)
    h2 = hash_user("urn:li:person:xyz789", config.hash_config)
    assert not np.array_equal(h1, h2)


def test_hash_range_user(config):
    h = hash_user("urn:li:person:test", config.hash_config)
    assert h.shape == (2,)
    assert np.all(h >= 1), "Hash values must be >= 1 (0 reserved for padding)"
    assert np.all(h < USER_TABLE_SIZE), f"Hash values must be < {USER_TABLE_SIZE}"


def test_hash_post_shape(config):
    h = hash_post("https://linkedin.com/posts/test-123", config.hash_config)
    assert h.shape == (3,)  # num_item_hashes=3
    assert np.all(h >= 1)
    assert np.all(h < POST_TABLE_SIZE)


def test_hash_post_deterministic(config):
    h1 = hash_post("https://linkedin.com/posts/test-123", config.hash_config)
    h2 = hash_post("https://linkedin.com/posts/test-123", config.hash_config)
    np.testing.assert_array_equal(h1, h2)


def test_hash_author(config):
    h = hash_author("john_doe", config.hash_config)
    assert h.shape == (2,)
    assert np.all(h >= 1)
    assert np.all(h < AUTHOR_TABLE_SIZE)


def test_hash_author_deterministic(config):
    h1 = hash_author("john_doe", config.hash_config)
    h2 = hash_author("john_doe", config.hash_config)
    np.testing.assert_array_equal(h1, h2)
