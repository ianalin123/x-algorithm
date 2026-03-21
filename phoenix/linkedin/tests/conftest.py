# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared pytest fixtures for LinkedIn Phoenix test suite."""

import pytest
import jax
import numpy as np

from linkedin.config import LinkedInModelConfig


@pytest.fixture
def example_config():
    """Returns a LinkedInModelConfig with default settings."""
    return LinkedInModelConfig()


@pytest.fixture
def rng_key():
    """Returns a JAX PRNGKey for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def example_batch(example_config):
    """Returns a LinkedInTrainingBatch with batch_size=1.

    NOTE: This fixture is intentionally lightweight — data.py must exist first.
    If data.py is not yet available, this fixture import will fail gracefully.
    """
    try:
        from linkedin.data import create_example_linkedin_batch

        return create_example_linkedin_batch(batch_size=1)
    except ImportError:
        pytest.skip("data.py not yet available")
