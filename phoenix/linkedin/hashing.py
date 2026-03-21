"""Feature hashing utilities for LinkedIn entity IDs.

Uses murmurhash3 (mmh3) for deterministic, cross-process-safe hashing.
Hash value 0 is reserved for padding/invalid entries.
"""

import numpy as np
import mmh3
from recsys_model import HashConfig

# Embedding table sizes
USER_TABLE_SIZE = 100_000
POST_TABLE_SIZE = 500_000
AUTHOR_TABLE_SIZE = 100_000

# Seed offsets for each hash function (must be distinct)
_HASH_SEEDS = [0, 1, 2, 3, 4, 5]


def hash_entity(
    entity_id: str,
    num_hashes: int,
    table_size: int,
    seed_offsets: list[int] | None = None,
) -> np.ndarray:
    """Hash a string entity ID to multiple embedding table indices.

    Args:
        entity_id: String identifier (e.g., URN, URL, username)
        num_hashes: Number of hash functions to apply
        table_size: Size of embedding table (indices in [1, table_size))
        seed_offsets: List of seed values for each hash function

    Returns:
        np.ndarray of shape [num_hashes] with dtype int32, values in [1, table_size)
    """
    if seed_offsets is None:
        seed_offsets = _HASH_SEEDS[:num_hashes]

    hashes = []
    for seed in seed_offsets[:num_hashes]:
        h = mmh3.hash(entity_id, seed=seed, signed=False)
        # Map to [1, table_size) — 0 reserved for padding
        idx = (h % (table_size - 1)) + 1
        hashes.append(idx)

    return np.array(hashes, dtype=np.int32)


def hash_user(profile_urn: str, config: HashConfig) -> np.ndarray:
    """Hash a user profile URN to embedding indices."""
    return hash_entity(profile_urn, config.num_user_hashes, USER_TABLE_SIZE)


def hash_post(post_url: str, config: HashConfig) -> np.ndarray:
    """Hash a post URL to embedding indices.

    Returns num_item_hashes values but only fills first 2 with real hashes.
    The 3rd slot is reserved for content embedding (will be filled separately).
    Non-hash slots are set to 1 (valid but meaningless — overwritten by content embedding).
    """
    num_real_hashes = min(2, config.num_item_hashes)
    real_hashes = hash_entity(post_url, num_real_hashes, POST_TABLE_SIZE)

    if config.num_item_hashes > num_real_hashes:
        # Pad with 1s for remaining slots (content embedding slots)
        padding = np.ones(config.num_item_hashes - num_real_hashes, dtype=np.int32)
        return np.concatenate([real_hashes, padding])

    return real_hashes


def hash_author(creator_username: str, config: HashConfig) -> np.ndarray:
    """Hash an author username to embedding indices."""
    return hash_entity(creator_username, config.num_author_hashes, AUTHOR_TABLE_SIZE)
