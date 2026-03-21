"""Profile feature embedder for cold-start LinkedIn users.

Converts LinkedIn profile attributes into a 128-dimensional embedding vector
for users with fewer than 10 interactions, where engagement history is
unavailable. The resulting embedding fills the user_embeddings slot in the
ranking model.

Feature layout (70d input → 128d output):
  - headline: text-embedding-3-small → projected to 64d
  - follower_count: log1p normalized → 1d
  - connection_count: log1p normalized → 1d
  - location_country: mmh3 hash → 1d
  - has_education: binary → 1d
  - has_work_experience: binary → 1d
  - spare: zero → 1d

The 70→128 projection uses a fixed random Gaussian matrix (seed=43).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import mmh3
import numpy as np

logger = logging.getLogger(__name__)

HEADLINE_EMB_DIM = 64
NUMERICAL_DIM = 6  # follower, connection, location, education, work_exp + 1 spare
PROFILE_INPUT_DIM = HEADLINE_EMB_DIM + NUMERICAL_DIM
PROFILE_OUTPUT_DIM = 128


class ProfileFeatureEmbedder:
    """Embeds LinkedIn profile attributes into a fixed-size vector.

    Used for cold-start users (< 10 reactions) where engagement history
    is unavailable. The resulting embedding fills the user_embeddings slot.
    """

    def __init__(
        self,
        cache_dir: str = ".embedding_cache",
        api_key: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._projection_matrix: Optional[np.ndarray] = None
        self._embedding_service = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from linkedin.embeddings import ContentEmbeddingService

            self._embedding_service = ContentEmbeddingService(
                cache_dir=str(self.cache_dir),
                target_dim=HEADLINE_EMB_DIM,
                api_key=self._api_key,
            )
        return self._embedding_service

    def _get_projection_matrix(self) -> np.ndarray:
        if self._projection_matrix is not None:
            return self._projection_matrix

        proj_path = self.cache_dir / "profile_projection_matrix.npy"
        if proj_path.exists():
            matrix = np.asarray(np.load(str(proj_path)), dtype=np.float32)
            self._projection_matrix = matrix
            return matrix

        rng = np.random.default_rng(seed=43)
        matrix = rng.normal(
            0,
            1.0 / np.sqrt(PROFILE_OUTPUT_DIM),
            size=(PROFILE_INPUT_DIM, PROFILE_OUTPUT_DIM),
        ).astype(np.float32)
        np.save(str(proj_path), matrix)
        self._projection_matrix = matrix
        return matrix

    def _encode_headline(self, headline: Optional[str]) -> np.ndarray:
        """Embed headline text and project to HEADLINE_EMB_DIM."""
        if not headline or not headline.strip():
            return np.zeros(HEADLINE_EMB_DIM, dtype=np.float32)

        try:
            svc = self._get_embedding_service()
            raw_emb = svc.embed_text(headline)  # [1536]
            projected = svc.project_to_model_dim(raw_emb, target_dim=HEADLINE_EMB_DIM)
            return projected.astype(np.float32)
        except (ValueError, Exception):
            logger.warning("Failed to embed headline, using zero vector")
            return np.zeros(HEADLINE_EMB_DIM, dtype=np.float32)

    def _encode_numerical(self, profile: dict) -> np.ndarray:
        """Encode numerical/categorical profile features into 6d vector."""
        features = np.zeros(NUMERICAL_DIM, dtype=np.float32)

        follower_count = profile.get("follower_count") or 0
        features[0] = float(np.log1p(max(0, follower_count))) / 15.0

        connection_count = profile.get("connection_count") or 0
        features[1] = float(np.log1p(max(0, connection_count))) / 10.0

        location_country = profile.get("location_country") or ""
        if location_country:
            features[2] = float(mmh3.hash(location_country, signed=False) % 10000) / 10000.0

        features[3] = float(bool(profile.get("has_education", False)))
        features[4] = float(bool(profile.get("has_work_experience", False)))
        # features[5] = 0 (spare)

        return features

    def embed_profile(self, profile: dict) -> np.ndarray:
        """Embed a single LinkedIn profile into a 128d vector.

        Args:
            profile: Dict with optional keys: headline, follower_count,
                     connection_count, location_country, has_education,
                     has_work_experience

        Returns:
            np.ndarray of shape [128]
        """
        headline_emb = self._encode_headline(profile.get("headline"))
        numerical_emb = self._encode_numerical(profile)

        combined = np.concatenate([headline_emb, numerical_emb], axis=0)  # [70]
        proj = self._get_projection_matrix()  # [70, 128]
        result = (combined @ proj).astype(np.float32)
        return result

    def embed_profiles_batch(self, profiles: list) -> np.ndarray:
        """Embed a batch of profiles.

        Returns:
            np.ndarray of shape [N, 128]
        """
        return np.stack([self.embed_profile(p) for p in profiles], axis=0)
