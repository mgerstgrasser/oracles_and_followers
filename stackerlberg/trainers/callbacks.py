import argparse
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker


class PolicyIntoEnv(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """A callback that puts the policies into the env, so that the env can access them.
        Used for hidden-queries experiments, similar to the two callbacks below.
        All three implement the same behavior in different ways."""
        for env in base_env.envs:
            env.unwrapped.policies = policies
            env.unwrapped.worker = worker


class DeleteHiddenQueriesCallback(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs,
    ):
        """A callback that deletes hidden queries from the batch.
        WARNING! This is hacky, and will *only* work if the hidden queries are the first items in the batch.
        In particular, only use this if the env puts them at the start of the episode only, *and* make sure
        that each batch is exactly one episode long, i.e. batch_mode="complete_episodes" and rollout_fragment_length=1."""
        first_real_step = -1
        for i in range(postprocessed_batch.count):
            if "hidden" in postprocessed_batch["infos"][i] and postprocessed_batch["infos"][i]["hidden"]:
                first_real_step = i
        if first_real_step != -1:
            test_postprocessed_batch = postprocessed_batch.slice(first_real_step + 2, postprocessed_batch.count)
            for key in postprocessed_batch:
                postprocessed_batch[key] = postprocessed_batch[key][first_real_step + 2 :]
            postprocessed_batch.count = postprocessed_batch.count - first_real_step - 2

        pass


class DeleteHiddenQueriesPrePostprocessCallback(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        pre_postprocess: bool = False,
        **kwargs,
    ):
        """A callback that deletes hidden queries from the batch.
        WARNING! This is hacky, and will *only* work if the hidden queries are the first items in the batch.
        In particular, only use this if the env puts them at the start of the episode only, *and* make sure
        that each batch is exactly one episode long, i.e. batch_mode="complete_episodes" and rollout_fragment_length=1."""
        if pre_postprocess:
            first_real_step = -1
            for i in range(postprocessed_batch.count):
                if "hidden" in postprocessed_batch["infos"][i] and postprocessed_batch["infos"][i]["hidden"]:
                    first_real_step = i
            if first_real_step != -1:
                truncated_batch = postprocessed_batch.slice(first_real_step + 2, postprocessed_batch.count)
                original_batches[policy_id] = (policies[policy_id], truncated_batch)
                for key in postprocessed_batch:
                    postprocessed_batch[key] = postprocessed_batch[key][first_real_step + 2 :]
                postprocessed_batch.count = postprocessed_batch.count - first_real_step - 2
        else:
            pass

        pass
