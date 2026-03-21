import numpy as np
import pandas as pd

from linkedin.config import REACTION_TYPE_MAP, LinkedInModelConfig, LINKEDIN_ACTIONS
from linkedin.data_pipeline import (
    UserHistoryBuilder,
    NegativeSampler,
    LinkedInBatchAssembler,
)


def make_reactions_df(n_users=3, n_reactions_per_user=5):
    rows = []
    base_time = pd.Timestamp("2024-06-01")
    reaction_types = list(REACTION_TYPE_MAP.keys())
    for user_index in range(n_users):
        for reaction_index in range(n_reactions_per_user):
            rows.append(
                {
                    "provider_profile_urn": f"urn:li:person:user{user_index}",
                    "provider_post_urn": f"urn:li:activity:post{user_index * 100 + reaction_index}",
                    "type": reaction_types[reaction_index % len(reaction_types)],
                    "reacted_at": base_time + pd.Timedelta(days=reaction_index),
                }
            )
    return pd.DataFrame(rows)


def make_posts_df(n_posts=20):
    rows = []
    base_time = pd.Timestamp("2024-05-25")
    for post_index in range(n_posts):
        rows.append(
            {
                "post_url": f"https://linkedin.com/posts/post{post_index}",
                "provider_urn": f"urn:li:activity:post{post_index}",
                "post_text": f"Post text {post_index}",
                "creator_username": f"author{post_index % 5}",
                "post_type": ["text", "image", "video"][post_index % 3],
                "posted_at": base_time + pd.Timedelta(days=post_index),
                "total_reactions": 100 + post_index,
                "total_comments": 10 + post_index,
                "total_reposts": 5,
            }
        )
    return pd.DataFrame(rows)


def test_history_chronological():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=5)
    builder = UserHistoryBuilder(max_history_len=32)
    histories = builder.build_histories(reactions)

    assert len(histories) == 1
    history = list(histories.values())[0]

    for index in range(len(history.timestamps) - 1):
        assert history.timestamps[index] <= history.timestamps[index + 1]


def test_history_action_vectors():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=3)
    builder = UserHistoryBuilder(max_history_len=32)
    histories = builder.build_histories(reactions)

    history = list(histories.values())[0]

    for index in range(history.length):
        assert history.action_vectors[index].sum() == 1.0
        assert np.argmax(history.action_vectors[index]) < 6


def test_history_max_length():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=50)
    builder = UserHistoryBuilder(max_history_len=10)
    histories = builder.build_histories(reactions)

    history = list(histories.values())[0]
    assert history.length <= 10


def test_density_stats():
    reactions = make_reactions_df(n_users=5, n_reactions_per_user=8)
    builder = UserHistoryBuilder()
    histories = builder.build_histories(reactions)
    stats = builder.get_density_stats(histories)

    assert stats["total_users"] == 5
    assert stats["sparse_lt10"] == 5


def test_negative_ratio():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=3)
    posts = make_posts_df(n_posts=50)

    sampler = NegativeSampler(negative_ratio=5, seed=42)
    pairs = sampler.build_training_pairs(reactions, posts)

    positives = pairs[pairs["label"] == 1]
    negatives = pairs[pairs["label"] == 0]

    assert len(positives) == 3
    assert len(negatives) <= 3 * 5
    assert len(negatives) > 0


def test_no_false_negatives():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=3)
    posts = make_posts_df(n_posts=50)

    sampler = NegativeSampler(negative_ratio=5, seed=42)
    pairs = sampler.build_training_pairs(reactions, posts)

    user_urn = reactions.iloc[0]["provider_profile_urn"]
    positive_urns = set(
        reactions[reactions["provider_profile_urn"] == user_urn]["provider_post_urn"]
    )

    negatives = pairs[(pairs["label"] == 0) & (pairs["user_urn"] == user_urn)]
    for _, row in negatives.iterrows():
        assert row["post_urn"] not in positive_urns


def test_batch_shapes():
    reactions = make_reactions_df(n_users=2, n_reactions_per_user=5)
    posts = make_posts_df(n_posts=250)

    config = LinkedInModelConfig()
    builder = UserHistoryBuilder()
    histories = builder.build_histories(reactions, posts)

    sampler = NegativeSampler(negative_ratio=2, seed=42)
    pairs = sampler.build_training_pairs(reactions, posts)

    assembler = LinkedInBatchAssembler(config)
    batch = assembler.assemble_batch(pairs, histories, posts, list(range(min(4, len(pairs)))))

    batch_size = min(4, len(pairs))
    history_len = config.history_seq_len

    assert batch.batch.history_actions.shape == (batch_size, history_len, len(LINKEDIN_ACTIONS))
    assert batch.labels.reaction_labels.shape == (batch_size, 1, 6)
    assert batch.labels.comment_label.shape == (batch_size, 1)


def test_cold_start_user():
    posts = make_posts_df(n_posts=10)
    config = LinkedInModelConfig()

    pairs = pd.DataFrame(
        [
            {
                "user_urn": "urn:li:person:cold_start_user",
                "post_urn": "urn:li:activity:post0",
                "label": 0,
                "reaction_type": "",
            }
        ]
    )

    assembler = LinkedInBatchAssembler(config)
    batch = assembler.assemble_batch(pairs, {}, posts, [0])

    assert np.all(batch.batch.history_actions == 0)
    assert batch.batch.user_hashes.shape == (1, 2)


def test_data_leakage_prevention():
    reactions = make_reactions_df(n_users=1, n_reactions_per_user=5)
    posts = make_posts_df(n_posts=250)

    config = LinkedInModelConfig()
    builder = UserHistoryBuilder()
    histories = builder.build_histories(reactions, posts)

    target_post = reactions.iloc[0]["provider_post_urn"]
    pairs = pd.DataFrame(
        [
            {
                "user_urn": str(reactions.iloc[0]["provider_profile_urn"]),
                "post_urn": str(target_post),
                "label": 1,
                "reaction_type": "LIKE",
            }
        ]
    )

    assembler = LinkedInBatchAssembler(config)
    batch = assembler.assemble_batch(pairs, histories, posts, [0])

    from linkedin.hashing import hash_post

    target_hash = hash_post(str(target_post), config.hash_config)
    hist_hashes = batch.batch.history_post_hashes[0]

    for slot in range(hist_hashes.shape[0]):
        if np.all(hist_hashes[slot] == 0):
            continue
        assert not np.array_equal(hist_hashes[slot, :2], target_hash[:2])
