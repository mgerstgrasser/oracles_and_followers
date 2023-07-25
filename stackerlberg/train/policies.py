import random

import numpy as np
import tree  # pip install dm_tree
from gym.spaces import Box
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights


class ConstantSettablePolicy(Policy):
    """Hand-coded policy that returns constant actions, which can be set using set_weights()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        return [self.cur_action for _ in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {"action": self.cur_action}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        self.cur_action = weights["action"]

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )


class MatrixCoordGameBestResponsePolicy(Policy):
    """Hand-coded policy that returns constant actions, which can be set using set_weights()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.
        if not "none_0" in obs_batch:
            obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
            return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )


def IPD_MostlyTFT(prob):
    class IPDProbTFT(Policy):
        """Hand-coded policy that plays `prob` TFT, `1-prob` defect in iterated prisoner's dilemma.
        For prob >= 0.75 this should be the optimal leader policy,
        follower best-response should be cooperate always."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Whether for compute_actions, the bounds given in action_space
            # should be ignored (default: False). This is to test action-clipping
            # and any Env's reaction to bounds breaches.
            if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
                self.action_space_for_sampling = Box(
                    -float("inf"),
                    float("inf"),
                    shape=self.action_space.shape,
                    dtype=self.action_space.dtype,
                )
            else:
                self.action_space_for_sampling = self.action_space
            self.cur_action = self.action_space_for_sampling.sample()

        @override(Policy)
        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
            # Alternatively, a numpy array would work here as well.
            # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
            # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.

            # if np.random.choice([True, False], p=[prob, 1 - prob]):
            #     # TFT
            if isinstance(self.observation_space, Box):
                obs_batch = [np.argmax(obs) for obs in obs_batch]

            return (
                [(0 if obs <= 2 else 1) if np.random.choice([True, False], p=[prob, 1 - prob]) else 1 for obs in obs_batch],
                [],
                {},
            )
            # else:
            #     # Defect
            #     return [1 for _ in obs_batch], [], {}

            # if not "none_0" in obs_batch:
            #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
            #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
            # return [obs for obs in obs_batch["none_0"]], [], {}

        @override(Policy)
        def learn_on_batch(self, samples):
            """No learning."""
            return {}

        @override(Policy)
        def compute_log_likelihoods(
            self,
            actions,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
        ):
            return np.array([random.random()] * len(obs_batch))

        @override(Policy)
        def get_weights(self) -> ModelWeights:
            """No weights to save."""
            return {}

        @override(Policy)
        def set_weights(self, weights: ModelWeights) -> None:
            """No weights to set."""
            pass

        @override(Policy)
        def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
            return SampleBatch(
                {
                    SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
                }
            )

        def get_initial_state(self):
            """Returns initial RNN state for the current policy.

            Returns:
                List[TensorType]: Initial RNN state for the current policy.
            """
            return []

    return IPDProbTFT


def IPD_TFT_Coop_Defect():
    class IPDProbTFT(Policy):
        """Hand-coded policy that plays `prob` TFT, `1-prob` defect in iterated prisoner's dilemma.
        For prob >= 0.75 this should be the optimal leader policy,
        follower best-response should be cooperate always."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Whether for compute_actions, the bounds given in action_space
            # should be ignored (default: False). This is to test action-clipping
            # and any Env's reaction to bounds breaches.
            if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
                self.action_space_for_sampling = Box(
                    -float("inf"),
                    float("inf"),
                    shape=self.action_space.shape,
                    dtype=self.action_space.dtype,
                )
            else:
                self.action_space_for_sampling = self.action_space
            self.cur_action = self.action_space_for_sampling.sample()

        @override(Policy)
        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
            state_batches = state_batches[0]
            # If the current state batch is 0, we randomise which policy we play this episode.
            state_batches = np.array([[np.random.randint(0, 4) if q[0] == -1 else q[0]] for q in state_batches])
            # Alternatively, a numpy array would work here as well.
            # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
            # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.

            # if np.random.choice([True, False], p=[prob, 1 - prob]):
            #     # TFT
            if isinstance(self.observation_space, Box):
                obs_batch = [np.argmax(obs) for obs in obs_batch]

            actions = [
                lambda obs: 0,  # always coop
                lambda obs: 1,  # always defect
                lambda obs: 0 if obs <= 2 else 1,  # TFT
                lambda obs: 0 if 1 <= obs <= 2 else 1,  # unfriendly TFT
            ]

            return [actions[state_batches[i][0]](obs_batch[i]) for i in range(len(obs_batch))], [state_batches], {}
            # else:
            #     # Defect
            #     return [1 for _ in obs_batch], [], {}

            # if not "none_0" in obs_batch:
            #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
            #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
            # return [obs for obs in obs_batch["none_0"]], [], {}

        @override(Policy)
        def learn_on_batch(self, samples):
            """No learning."""
            return {}

        @override(Policy)
        def compute_log_likelihoods(
            self,
            actions,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
        ):
            return np.array([random.random()] * len(obs_batch))

        @override(Policy)
        def get_weights(self) -> ModelWeights:
            """No weights to save."""
            return {}

        @override(Policy)
        def set_weights(self, weights: ModelWeights) -> None:
            """No weights to set."""
            pass

        @override(Policy)
        def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
            return SampleBatch(
                {
                    SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
                }
            )

        def get_initial_state(self):
            """Returns initial RNN state for the current policy.

            Returns:
                List[TensorType]: Initial RNN state for the current policy.
            """
            return np.array([[-1]])

    return IPDProbTFT


class IPDRandomEveryEpisodePolicy(Policy):
    """Hand-coded policy that plays `prob` TFT, `1-prob` defect in iterated prisoner's dilemma.
    For prob >= 0.75 this should be the optimal leader policy,
    follower best-response should be cooperate always."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        pass
        # state_batches = state_batches[0]
        # If the current state batch is 0, we randomise which policy we play this episode.
        state_batches = [np.array([np.random.randint(2) if q[0][i] == -1 else q[0][i] for i in range(len(q[0]))]) for q in state_batches]
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.

        # if np.random.choice([True, False], p=[prob, 1 - prob]):
        #     # TFT
        if isinstance(self.observation_space, Box):
            obs_batch = [np.argmax(obs) for obs in obs_batch]

        return [state_batches[i][obs_batch[i]] for i in range(len(obs_batch))], [state_batches], {}
        # else:
        #     # Defect
        #     return [1 for _ in obs_batch], [], {}

        # if not "none_0" in obs_batch:
        #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
        #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        # return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )

    def get_initial_state(self):
        """Returns initial RNN state for the current policy.

        Returns:
            List[TensorType]: Initial RNN state for the current policy.
        """
        return [np.array([-1, -1, -1, -1, -1])]


class AlwaysCoop(Policy):
    """Hand-coded policy that always cooperates in IPD."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.
        return [0 for _ in obs_batch], [], {}

        # if not "none_0" in obs_batch:
        #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
        #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        # return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )


class AlwaysDefect(Policy):
    """Hand-coded policy that always defects in IPD."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.
        return [1 for _ in obs_batch], [], {}

        # if not "none_0" in obs_batch:
        #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
        #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        # return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )


class IPDCoopOrDefectPerEpisode(Policy):
    """Hand-coded policy that plays `prob` TFT, `1-prob` defect in iterated prisoner's dilemma.
    For prob >= 0.75 this should be the optimal leader policy,
    follower best-response should be cooperate always."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        state_batches = state_batches[0]
        # If the current state batch is 0, we randomise which policy we play this episode.
        state_batches = np.array([[np.random.randint(0, 2) if q[0] == -1 else q[0]] for q in state_batches])
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.

        # if np.random.choice([True, False], p=[prob, 1 - prob]):
        #     # TFT
        if isinstance(self.observation_space, Box):
            obs_batch = [np.argmax(obs) for obs in obs_batch]

        actions = [
            lambda obs: 0,  # always coop
            lambda obs: 1,  # always defect
            # lambda obs: 0 if obs <= 2 else 1,  # TFT
            # lambda obs: 0 if 1 <= obs <= 2 else 1,  # unfriendly TFT
        ]

        return [actions[state_batches[i][0]](obs_batch[i]) for i in range(len(obs_batch))], [state_batches], {}
        # else:
        #     # Defect
        #     return [1 for _ in obs_batch], [], {}

        # if not "none_0" in obs_batch:
        #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
        #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        # return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )

    def get_initial_state(self):
        """Returns initial RNN state for the current policy.

        Returns:
            List[TensorType]: Initial RNN state for the current policy.
        """
        return np.array([[-1]])


class SmIPD_TFT_Coop_Defect(Policy):
    """Hand-coded policy that plays `prob` TFT, `1-prob` defect in iterated prisoner's dilemma.
    For prob >= 0.75 this should be the optimal leader policy,
    follower best-response should be cooperate always."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space
        self.cur_action = self.action_space_for_sampling.sample()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        state_batches = state_batches[0]
        # If the current state batch is 0, we randomise which policy we play this episode.
        state_batches = np.array([[np.random.randint(0, 4) if q[0] == -1 else q[0]] for q in state_batches])
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # The following is meant to handle cases for different learning algorithms. Some of them flatten the obs, some don't.

        # if np.random.choice([True, False], p=[prob, 1 - prob]):
        #     # TFT
        if isinstance(self.observation_space, Box):
            obs_batch = [np.argmax(obs) for obs in obs_batch]

        actions = [
            lambda obs: 0,  # always coop
            lambda obs: 1,  # always defect
            lambda obs: 0 if obs <= 1 else 1,  # TFT
            lambda obs: 0 if obs == 1 else 1,  # unfriendly TFT
        ]

        return [actions[state_batches[i][0]](obs_batch[i]) for i in range(len(obs_batch))], [state_batches], {}
        # else:
        #     # Defect
        #     return [1 for _ in obs_batch], [], {}

        # if not "none_0" in obs_batch:
        #     obs_batch = restore_original_dimensions(np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np)
        #     return [1 if obs[0] == 1.0 else 0 for obs in obs_batch["none_0"]], [], {}
        # return [obs for obs in obs_batch["none_0"]], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.cur_action),
            }
        )

    def get_initial_state(self):
        """Returns initial RNN state for the current policy.

        Returns:
            List[TensorType]: Initial RNN state for the current policy.
        """
        return np.array([[-1]])
