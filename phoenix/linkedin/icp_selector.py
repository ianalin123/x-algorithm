"""ICP audience selector — queries Pinecone for matching profiles and loads them from postgres."""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ICPAudienceSelector:
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        pinecone_index: Optional[str] = None,
        db_dsn: Optional[str] = None,
    ):
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.pinecone_index_name = pinecone_index or os.environ.get("PINECONE_INDEX")
        self.db_dsn = db_dsn or os.environ.get("LINKEDIN_DB_DSN")
        self._pinecone_index = None
        self._db_loader = None

    def _get_pinecone_index(self):
        if self._pinecone_index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.pinecone_api_key)
            self._pinecone_index = pc.Index(self.pinecone_index_name)
        return self._pinecone_index

    def _get_db_loader(self):
        if self._db_loader is None:
            from linkedin.data_pipeline import LinkedInDataLoader

            self._db_loader = LinkedInDataLoader(dsn=self.db_dsn)
        return self._db_loader

    def select_audience(self, icp_description: str, max_profiles: int = 200) -> list[dict]:
        """Select profiles matching an ICP description via Pinecone KNN.

        1. Embed ICP description
        2. Query Pinecone for similar profiles
        3. Fetch full profile data from postgres
        """
        from linkedin.embeddings import ContentEmbeddingService

        svc = ContentEmbeddingService()
        icp_embedding = svc.embed_text(icp_description)

        index = self._get_pinecone_index()
        results = index.query(
            vector=icp_embedding.tolist(),
            top_k=max_profiles,
            filter={"record_type": {"$ne": "icp"}},
            include_metadata=True,
        )

        profile_urns = []
        for match in results.get("matches", []):
            urn = match.get("metadata", {}).get("profile_urn")
            if urn:
                profile_urns.append(urn)

        if not profile_urns:
            logger.warning(f"No profiles matched ICP: {icp_description[:50]}...")
            return []

        loader = self._get_db_loader()
        profiles_df = loader.load_profiles()
        matched = profiles_df[profiles_df["provider_urn"].isin(profile_urns)]

        return matched.to_dict("records")

    def select_overall_audience(self, max_profiles: int = 500) -> list[dict]:
        """Select a representative sample from the full dataset.

        Stratified by follower_count to ensure diversity.
        """
        loader = self._get_db_loader()
        profiles_df = loader.load_profiles()

        if len(profiles_df) <= max_profiles:
            return profiles_df.to_dict("records")

        profiles_df = profiles_df.copy()
        profiles_df["_bucket"] = "low"
        if "follower_count" in profiles_df.columns:
            profiles_df.loc[profiles_df["follower_count"] >= 1000, "_bucket"] = "mid"
            profiles_df.loc[profiles_df["follower_count"] >= 10000, "_bucket"] = "high"

        samples = []
        per_bucket = max_profiles // 3

        for bucket in ["low", "mid", "high"]:
            bucket_df = profiles_df[profiles_df["_bucket"] == bucket]
            n = min(per_bucket, len(bucket_df))
            if n > 0:
                samples.append(bucket_df.sample(n=n, random_state=42))

        result = profiles_df.iloc[0:0]
        if samples:
            import pandas as pd

            result = pd.concat(samples, ignore_index=True)

        remaining = max_profiles - len(result)
        if remaining > 0:
            unused = profiles_df[~profiles_df.index.isin(result.index)]
            extra = unused.sample(n=min(remaining, len(unused)), random_state=42)
            import pandas as pd

            result = pd.concat([result, extra], ignore_index=True)

        result = result.drop(columns=["_bucket"], errors="ignore")
        return result.to_dict("records")
