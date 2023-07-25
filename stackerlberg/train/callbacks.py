import argparse
import os
from typing import Dict, Tuple

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class InsertEpisodeEndsCallback(DefaultCallbacks):
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
        for i in range(postprocessed_batch.count):
            if "episode_done" in postprocessed_batch["infos"][i] and postprocessed_batch["infos"][i]["episode_done"] is True:
                postprocessed_batch["dones"][i] = True
