"""Content embedding service for LinkedIn posts.

Uses OpenAI text-embedding-3-small to embed post text.
Embeddings (1536d) are cached to disk and projected to model dimension (128d)
using a frozen random orthogonal projection matrix.

The projection matrix is:
- Computed once and saved to {cache_dir}/projection_matrix.npy
- NOT part of the model checkpoint (fixed preprocessing step)
- Applied at batch assembly time before feeding into the model
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# OpenAI embedding dimension for text-embedding-3-small
OPENAI_EMBEDDING_DIM = 1536
# Max texts per API batch
OPENAI_BATCH_SIZE = 2048


class ContentEmbeddingService:
    """Service for embedding text content using OpenAI text-embedding-3-small.

    Provides file-based caching and dimension projection.
    """

    def __init__(
        self,
        cache_dir: str = ".embedding_cache",
        target_dim: int = 128,
        api_key: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_dim = target_dim
        self._projection_matrix: Optional[np.ndarray] = None

        # Initialize OpenAI client
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            self._client = OpenAI(api_key=key)
        else:
            self._client = None
            logger.warning("OPENAI_API_KEY not set — API calls will fail")

    def _cache_key(self, text: str) -> str:
        """Generate cache filename from text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text)}.npy"

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text, using cache if available.

        Args:
            text: Text to embed. Empty string returns zero vector.

        Returns:
            np.ndarray of shape [1536]
        """
        if not text or not text.strip():
            return np.zeros(OPENAI_EMBEDDING_DIM, dtype=np.float32)

        cache_path = self._cache_path(text)
        if cache_path.exists():
            return np.load(str(cache_path))

        if self._client is None:
            raise ValueError("OpenAI client not initialized — set OPENAI_API_KEY")

        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        np.save(str(cache_path), embedding)
        return embedding

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts, using cache where available.

        Args:
            texts: List of texts to embed

        Returns:
            np.ndarray of shape [N, 1536]
        """
        results: list[Optional[np.ndarray]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(np.zeros(OPENAI_EMBEDDING_DIM, dtype=np.float32))
                continue

            cache_path = self._cache_path(text)
            if cache_path.exists():
                results.append(np.load(str(cache_path)))
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch API calls for uncached texts
        if uncached_texts and self._client is not None:
            for batch_start in range(0, len(uncached_texts), OPENAI_BATCH_SIZE):
                batch = uncached_texts[batch_start : batch_start + OPENAI_BATCH_SIZE]
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                for j, data in enumerate(response.data):
                    idx = uncached_indices[batch_start + j]
                    text = uncached_texts[batch_start + j]
                    embedding = np.array(data.embedding, dtype=np.float32)
                    np.save(str(self._cache_path(text)), embedding)
                    results[idx] = embedding

        return np.array(results, dtype=np.float32)

    def get_projection_matrix(self) -> np.ndarray:
        """Get or create the frozen 1536→target_dim projection matrix.

        Matrix is saved to disk on first creation and reloaded on subsequent calls.
        Uses random Gaussian projection (Johnson-Lindenstrauss).

        Returns:
            np.ndarray of shape [1536, target_dim]
        """
        if self._projection_matrix is not None:
            return self._projection_matrix

        proj_path = self.cache_dir / "projection_matrix.npy"
        if proj_path.exists():
            loaded: np.ndarray = np.load(str(proj_path))
            self._projection_matrix = loaded
            return loaded

        # Create random Gaussian projection matrix
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        matrix = rng.normal(
            0,
            1.0 / np.sqrt(self.target_dim),
            size=(OPENAI_EMBEDDING_DIM, self.target_dim),
        ).astype(np.float32)

        np.save(str(proj_path), matrix)
        self._projection_matrix = matrix
        logger.info(f"Created projection matrix: {OPENAI_EMBEDDING_DIM}→{self.target_dim}")
        return matrix

    def project_to_model_dim(
        self,
        embeddings: np.ndarray,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Project embeddings from 1536d to model dimension using frozen matrix.

        Args:
            embeddings: np.ndarray of shape [N, 1536] or [1536]

        Returns:
            np.ndarray of shape [N, target_dim] or [target_dim]
        """
        proj = self.get_projection_matrix()

        if embeddings.ndim == 1:
            if np.all(embeddings == 0):
                return np.zeros(self.target_dim, dtype=np.float32)
            return (embeddings @ proj).astype(np.float32)

        return (embeddings @ proj).astype(np.float32)

    def precompute_all(
        self,
        posts_df,
        text_column: str = "post_text",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Precompute and cache embeddings for all posts in DataFrame.

        Args:
            posts_df: DataFrame with post text column
            text_column: Name of the text column
            cache_dir: Optional override for cache directory
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        texts = posts_df[text_column].fillna("").tolist()
        logger.info(f"Precomputing {len(texts)} post embeddings...")

        total = len(texts)
        for i in range(0, total, OPENAI_BATCH_SIZE):
            batch = texts[i : i + OPENAI_BATCH_SIZE]
            self.embed_texts(batch)
            logger.info(f"Embedded {min(i + OPENAI_BATCH_SIZE, total)}/{total} posts")
