"""LinkedIn recommendation system configuration.

Defines actions, action types, post types, and model configs for LinkedIn ranking.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

import jax.numpy as jnp

from grok import TransformerConfig
from recsys_model import PhoenixModelConfig, HashConfig


# Action definitions
LINKEDIN_ACTIONS = [
    "like_score",
    "celebrate_score",
    "support_score",
    "love_score",
    "insightful_score",
    "funny_score",
    "comment_score",
    "repost_score",
]

# First 6 are mutually exclusive reactions
LINKEDIN_REACTION_ACTIONS = LINKEDIN_ACTIONS[:6]

# Last 2 are independent actions
LINKEDIN_INDEPENDENT_ACTIONS = LINKEDIN_ACTIONS[6:]


# LinkedIn API reaction type to action index mapping
REACTION_TYPE_MAP = {
    "LIKE": 0,
    "PRAISE": 1,
    "EMPATHY": 2,
    "APPRECIATION": 3,
    "INTEREST": 4,
    "ENTERTAINMENT": 5,
}

# Post type to product_surface index mapping
POST_TYPE_MAP = {
    "text": 0,
    "image": 1,
    "video": 2,
    "document": 3,
    "article": 4,
    "poll": 5,
}


@dataclass
class LinkedInModelConfig:
    """Configuration for LinkedIn ranking model.

    Wraps PhoenixModelConfig with LinkedIn-specific defaults.
    """

    num_actions: int = 8
    history_seq_len: int = 32
    candidate_seq_len: int = 1
    product_surface_vocab_size: int = 6
    emb_size: int = 128

    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16

    hash_config: HashConfig = field(
        default_factory=lambda: HashConfig(
            num_user_hashes=2,
            num_item_hashes=3,
            num_author_hashes=2,
        )
    )

    transformer_config: Optional[TransformerConfig] = None

    def __post_init__(self):
        """Initialize transformer config with defaults if not provided."""
        if self.transformer_config is None:
            self.transformer_config = TransformerConfig(
                emb_size=self.emb_size,
                key_size=64,
                num_q_heads=2,
                num_kv_heads=2,
                num_layers=4,
                widening_factor=2,
                attn_output_multiplier=0.125,
            )

    def make(self) -> PhoenixModelConfig:
        """Create PhoenixModelConfig from LinkedIn config."""
        return PhoenixModelConfig(
            model=self.transformer_config,
            emb_size=self.emb_size,
            num_actions=self.num_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            hash_config=self.hash_config,
            product_surface_vocab_size=self.product_surface_vocab_size,
            name=self.name,
            fprop_dtype=self.fprop_dtype,
        ).initialize()
