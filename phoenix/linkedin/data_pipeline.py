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


if __name__ == "__main__":
    dsn = os.environ.get("LINKEDIN_DB_DSN")
    if not dsn:
        print("SKIP: LINKEDIN_DB_DSN not set")
        raise SystemExit(0)

    with LinkedInDataLoader(dsn) as loader:
        posts = loader.load_posts(limit=10)
        assert len(posts) == 10, f"Expected 10 posts, got {len(posts)}"
        print("CONNECTED")
