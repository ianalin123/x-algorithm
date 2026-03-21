"""LinkedIn data pipeline — loads data from postgres into pandas DataFrames."""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class LinkedInDataLoader:
    """Loads LinkedIn data from postgres.

    Provides methods for loading posts, profiles, and reactions.
    All queries use parameterized SQL to prevent injection.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ.get("LINKEDIN_DB_DSN")
        if not self.dsn:
            raise ValueError("LINKEDIN_DB_DSN not set — provide dsn or set env var")
        self._conn = None

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def load_posts(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load linkedin_posts table (pre-publish attributes only, no target columns)."""
        conn = self._get_connection()

        query = """
            SELECT
                post_text, post_url, provider_urn,
                creator_first_name, creator_last_name, creator_headline,
                creator_username, creator_profile_url,
                posted_at, post_type, has_video,
                attachments,
                total_reactions, total_comments, total_reposts,
                like_reactions, support_reactions, love_reactions,
                insight_reactions, celebrate_reactions, funny_reactions
            FROM linkedin_posts
            ORDER BY posted_at DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        return pd.read_sql_query(query, conn)

    def load_profiles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load linkedin_profiles table."""
        conn = self._get_connection()

        query = """
            SELECT
                username, provider_urn, first_name, last_name,
                headline, about, profile_url,
                follower_count, connection_count,
                location_full, location_country, location_city,
                has_education, has_work_experience
            FROM linkedin_profiles
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        return pd.read_sql_query(query, conn)

    def load_reactions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load linkedin_reactions table."""
        conn = self._get_connection()

        query = """
            SELECT
                provider_profile_urn, provider_post_urn,
                type, reacted_at
            FROM linkedin_reactions
            ORDER BY reacted_at ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        return pd.read_sql_query(query, conn, parse_dates=["reacted_at"])

    def get_temporal_split(
        self,
        reactions_df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[datetime, datetime]:
        """Compute temporal train/val/test split boundaries.

        Args:
            reactions_df: DataFrame with 'reacted_at' column
            train_ratio: Fraction of data for training (default 0.70)
            val_ratio: Fraction of data for validation (default 0.15)

        Returns:
            (train_cutoff, val_cutoff) datetime tuple
            - Data before train_cutoff: training set
            - Data between train_cutoff and val_cutoff: validation set
            - Data after val_cutoff: test set
        """
        if "reacted_at" not in reactions_df.columns:
            raise ValueError("reactions_df must have 'reacted_at' column")

        timestamps = reactions_df["reacted_at"].sort_values()
        n = len(timestamps)

        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        train_cutoff = timestamps.iloc[train_idx]
        val_cutoff = timestamps.iloc[val_idx]

        logger.info(
            f"Temporal split: train < {train_cutoff}, "
            f"val in [{train_cutoff}, {val_cutoff}), test >= {val_cutoff}"
        )
        logger.info(f"  Train: {train_idx} reactions ({train_ratio:.0%})")
        logger.info(f"  Val: {val_idx - train_idx} reactions ({val_ratio:.0%})")
        logger.info(f"  Test: {n - val_idx} reactions ({1 - train_ratio - val_ratio:.0%})")

        return train_cutoff, val_cutoff

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


from dataclasses import dataclass, field
from typing import cast

import numpy as np

from linkedin.config import REACTION_TYPE_MAP, LINKEDIN_ACTIONS
from linkedin.hashing import hash_user, hash_post, hash_author
from linkedin.data import (
    LinkedInTrainingBatch,
    LinkedInEmbeddings,
    LinkedInLabels,
    encode_post_type,
)
from recsys_model import RecsysBatch


@dataclass
class UserHistory:
    profile_urn: str
    post_urns: list[str] = field(default_factory=list)
    author_usernames: list[str] = field(default_factory=list)
    action_vectors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(LINKEDIN_ACTIONS)), dtype=np.float32)
    )
    post_types: list[str] = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.post_urns)


class UserHistoryBuilder:
    def __init__(self, max_history_len: int = 32):
        self.max_history_len = max_history_len

    def build_histories(
        self,
        reactions_df: pd.DataFrame,
        posts_df: pd.DataFrame | None = None,
    ) -> dict[str, UserHistory]:
        histories = {}

        grouped = reactions_df.groupby("provider_profile_urn")

        for user_urn, user_reactions in grouped:
            user_reactions = user_reactions.sort_values("reacted_at")
            if len(user_reactions) > self.max_history_len:
                user_reactions = user_reactions.tail(self.max_history_len)

            n = len(user_reactions)
            action_vectors = np.zeros((n, len(LINKEDIN_ACTIONS)), dtype=np.float32)

            post_urns = []
            author_usernames = []
            post_types_list = []
            timestamps = []

            for i, (_, row) in enumerate(user_reactions.iterrows()):
                post_urn = row["provider_post_urn"]
                post_urns.append(post_urn)

                if posts_df is not None and "post_url" in posts_df.columns:
                    matching = posts_df[posts_df["provider_urn"] == post_urn]
                    if len(matching) > 0:
                        author_usernames.append(matching.iloc[0].get("creator_username", "unknown"))
                        post_types_list.append(matching.iloc[0].get("post_type", "text"))
                    else:
                        author_usernames.append("unknown")
                        post_types_list.append("text")
                else:
                    author_usernames.append("unknown")
                    post_types_list.append("text")

                reaction_type = row.get("type", "")
                if reaction_type in REACTION_TYPE_MAP:
                    action_idx = REACTION_TYPE_MAP[reaction_type]
                    action_vectors[i, action_idx] = 1.0

                timestamps.append(row.get("reacted_at"))

            histories[str(user_urn)] = UserHistory(
                profile_urn=str(user_urn),
                post_urns=post_urns,
                author_usernames=author_usernames,
                action_vectors=action_vectors,
                post_types=post_types_list,
                timestamps=timestamps,
            )

        return histories

    def get_density_stats(self, histories: dict[str, UserHistory]) -> dict:
        lengths = [history.length for history in histories.values()]
        return {
            "total_users": len(histories),
            "sparse_lt10": sum(1 for length in lengths if length < 10),
            "medium_10_30": sum(1 for length in lengths if 10 <= length < 30),
            "dense_30_plus": sum(1 for length in lengths if length >= 30),
            "avg_history_len": np.mean(lengths) if lengths else 0,
            "max_history_len": max(lengths) if lengths else 0,
        }


class NegativeSampler:
    def __init__(self, window_days: int = 7, negative_ratio: int = 5, seed: int = 42):
        self.window_days = window_days
        self.negative_ratio = negative_ratio
        self.rng = np.random.default_rng(seed)

    def sample_negatives(
        self,
        user_urn: str,
        positive_post_urn: str,
        positive_timestamp: pd.Timestamp,
        all_posts_df: pd.DataFrame,
        user_positive_urns: set[str],
    ) -> list[str]:
        del user_urn
        window_start = positive_timestamp - pd.Timedelta(days=self.window_days)
        window_end = positive_timestamp + pd.Timedelta(days=self.window_days)

        mask = (
            (all_posts_df["posted_at"] >= window_start)
            & (all_posts_df["posted_at"] <= window_end)
            & (~all_posts_df["provider_urn"].isin(list(user_positive_urns)))
            & (all_posts_df["provider_urn"] != positive_post_urn)
        )
        candidates = all_posts_df.loc[mask, "provider_urn"].values

        if len(candidates) == 0:
            return []

        n_samples = min(self.negative_ratio, len(candidates))
        selected = self.rng.choice(candidates, size=n_samples, replace=False)
        return selected.tolist()

    def build_training_pairs(
        self,
        reactions_df: pd.DataFrame,
        posts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        user_positives: dict[str, set[str]] = {}
        for _, row in reactions_df.iterrows():
            urn = str(row["provider_profile_urn"])
            if urn not in user_positives:
                user_positives[urn] = set()
            user_positives[urn].add(str(row["provider_post_urn"]))

        pairs = []

        for _, row in reactions_df.iterrows():
            user_urn = str(row["provider_profile_urn"])
            post_urn = str(row["provider_post_urn"])
            timestamp = pd.Timestamp(str(row["reacted_at"]))
            reaction_type = row.get("type", "")

            if pd.isna(timestamp):
                continue
            timestamp = cast(pd.Timestamp, timestamp)

            pairs.append(
                {
                    "user_urn": user_urn,
                    "post_urn": post_urn,
                    "label": 1,
                    "reaction_type": reaction_type,
                }
            )

            negatives = self.sample_negatives(
                user_urn, post_urn, timestamp, posts_df, user_positives.get(user_urn, set())
            )
            for neg_urn in negatives:
                pairs.append(
                    {
                        "user_urn": user_urn,
                        "post_urn": neg_urn,
                        "label": 0,
                        "reaction_type": "",
                    }
                )

        return pd.DataFrame(pairs)


class LinkedInBatchAssembler:
    def __init__(self, config, embedding_cache_dir: str = ".embedding_cache"):
        self.config = config
        self.embedding_cache_dir = embedding_cache_dir
        self.emb_size = config.emb_size
        self.history_len = config.history_seq_len

    def assemble_batch(
        self,
        training_pairs: pd.DataFrame,
        histories: dict[str, UserHistory],
        posts_df: pd.DataFrame,
        batch_indices: list[int],
    ) -> LinkedInTrainingBatch:
        batch_size = len(batch_indices)
        candidate_seq_len = 1
        history_seq_len = self.history_len
        emb_size = self.emb_size

        user_hashes = np.zeros(
            (batch_size, self.config.hash_config.num_user_hashes), dtype=np.int32
        )
        hist_post_hashes = np.zeros(
            (batch_size, history_seq_len, self.config.hash_config.num_item_hashes), dtype=np.int32
        )
        hist_author_hashes = np.zeros(
            (batch_size, history_seq_len, self.config.hash_config.num_author_hashes), dtype=np.int32
        )
        hist_actions = np.zeros(
            (batch_size, history_seq_len, len(LINKEDIN_ACTIONS)), dtype=np.float32
        )
        hist_product_surface = np.zeros((batch_size, history_seq_len), dtype=np.int32)
        cand_post_hashes = np.zeros(
            (batch_size, candidate_seq_len, self.config.hash_config.num_item_hashes), dtype=np.int32
        )
        cand_author_hashes = np.zeros(
            (batch_size, candidate_seq_len, self.config.hash_config.num_author_hashes),
            dtype=np.int32,
        )
        cand_product_surface = np.zeros((batch_size, candidate_seq_len), dtype=np.int32)

        reaction_labels = np.zeros((batch_size, candidate_seq_len, 6), dtype=np.float32)
        comment_labels = np.zeros((batch_size, candidate_seq_len), dtype=np.float32)
        repost_labels = np.zeros((batch_size, candidate_seq_len), dtype=np.float32)

        for batch_position, pair_idx in enumerate(batch_indices):
            row = training_pairs.iloc[pair_idx]
            user_urn = row["user_urn"]
            post_urn = row["post_urn"]
            label = row["label"]
            reaction_type = row.get("reaction_type", "")

            user_hashes[batch_position] = hash_user(user_urn, self.config.hash_config)
            cand_post_hashes[batch_position, 0] = hash_post(post_urn, self.config.hash_config)

            post_match = posts_df[posts_df["provider_urn"] == post_urn]
            if len(post_match) > 0:
                post_row = post_match.iloc[0]
                author_username = post_row.get("creator_username", "unknown")
                post_type = post_row.get("post_type", "text")
                total_comments = post_row.get("total_comments", 0) or 0
                total_reactions = post_row.get("total_reactions", 0) or 0
                total_reposts = post_row.get("total_reposts", 0) or 0
            else:
                author_username = "unknown"
                post_type = "text"
                total_comments = 0
                total_reactions = 0
                total_reposts = 0

            cand_author_hashes[batch_position, 0] = hash_author(
                author_username, self.config.hash_config
            )
            cand_product_surface[batch_position, 0] = encode_post_type(post_type)

            history = histories.get(user_urn)
            if history is not None and history.length > 0:
                valid_indices = [
                    history_idx
                    for history_idx, urn in enumerate(history.post_urns)
                    if urn != post_urn
                ]
                num_valid = min(len(valid_indices), history_seq_len)

                for history_position, history_idx in enumerate(valid_indices[-num_valid:]):
                    hist_post_hashes[batch_position, history_position] = hash_post(
                        history.post_urns[history_idx], self.config.hash_config
                    )
                    hist_author_hashes[batch_position, history_position] = hash_author(
                        history.author_usernames[history_idx], self.config.hash_config
                    )
                    hist_actions[batch_position, history_position] = history.action_vectors[
                        history_idx
                    ]
                    hist_product_surface[batch_position, history_position] = encode_post_type(
                        history.post_types[history_idx]
                    )

            if label == 1 and reaction_type in REACTION_TYPE_MAP:
                reaction_labels[batch_position, 0, REACTION_TYPE_MAP[reaction_type]] = 1.0

            if label == 1 and total_reactions > 0:
                comment_labels[batch_position, 0] = min(
                    float(total_comments) / float(total_reactions), 1.0
                )
                repost_labels[batch_position, 0] = min(
                    float(total_reposts) / float(total_reactions), 1.0
                )

        batch = RecsysBatch(
            user_hashes=user_hashes,
            history_post_hashes=hist_post_hashes,
            history_author_hashes=hist_author_hashes,
            history_actions=hist_actions,
            history_product_surface=hist_product_surface,
            candidate_post_hashes=cand_post_hashes,
            candidate_author_hashes=cand_author_hashes,
            candidate_product_surface=cand_product_surface,
        )

        rng = np.random.default_rng(42)
        embeddings = LinkedInEmbeddings.from_components(
            user_embeddings=(
                rng.normal(
                    size=(batch_size, self.config.hash_config.num_user_hashes, emb_size)
                ).astype(np.float32)
                * 0.01
            ),
            history_post_embeddings=(
                rng.normal(
                    size=(
                        batch_size,
                        history_seq_len,
                        self.config.hash_config.num_item_hashes,
                        emb_size,
                    )
                ).astype(np.float32)
                * 0.01
            ),
            history_author_embeddings=(
                rng.normal(
                    size=(
                        batch_size,
                        history_seq_len,
                        self.config.hash_config.num_author_hashes,
                        emb_size,
                    )
                ).astype(np.float32)
                * 0.01
            ),
            candidate_author_embeddings=(
                rng.normal(
                    size=(
                        batch_size,
                        candidate_seq_len,
                        self.config.hash_config.num_author_hashes,
                        emb_size,
                    )
                ).astype(np.float32)
                * 0.01
            ),
            content_embeddings=np.zeros(
                (batch_size, candidate_seq_len, emb_size), dtype=np.float32
            ),
            emb_size=emb_size,
        )

        soft_comment_labels = comment_labels.copy()
        soft_repost_labels = repost_labels.copy()

        labels = LinkedInLabels(
            reaction_labels=reaction_labels,
            comment_label=(comment_labels > 0).astype(np.float32),
            repost_label=(repost_labels > 0).astype(np.float32),
        )
        labels.comment_label = soft_comment_labels
        labels.repost_label = soft_repost_labels

        return LinkedInTrainingBatch(
            batch=batch,
            embeddings=embeddings,
            labels=labels,
        )

    def iterate_batches(
        self,
        training_pairs: pd.DataFrame,
        histories: dict[str, UserHistory],
        posts_df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ):
        n = len(training_pairs)
        indices = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size].tolist()
            yield self.assemble_batch(training_pairs, histories, posts_df, batch_indices)


if __name__ == "__main__":
    dsn = os.environ.get("LINKEDIN_DB_DSN")
    if not dsn:
        print("SKIP: LINKEDIN_DB_DSN not set")
        raise SystemExit(0)

    with LinkedInDataLoader(dsn) as loader:
        posts = loader.load_posts(limit=10)
        assert len(posts) == 10, f"Expected 10 posts, got {len(posts)}"
        print("CONNECTED")
