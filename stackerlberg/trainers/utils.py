from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker


def randomise_leader_policy_each_episode(leader_policy_id: str = "agent_0", skip_bias: bool = False):
    # Callback to randomise the weights of the leader agent at the start of each episode
    class RandomisePolicy0(DefaultCallbacks):
        def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2],
            **kwargs,
        ) -> None:
            cur_weights = policies[leader_policy_id].get_weights()
            if isinstance(cur_weights, dict):
                for key in cur_weights:
                    if skip_bias and "bias" in key:
                        continue
                    cur_weights[key] = np.random.uniform(low=-1, high=1, size=cur_weights[key].shape)
            elif isinstance(cur_weights, list):
                if skip_bias:
                    print("Warning: skip_bias is not implemented for list weights")
                for i in range(len(cur_weights)):
                    cur_weights[i] = np.random.uniform(low=-1, high=1, size=cur_weights[i].shape)
            else:
                raise ValueError("Weights are neither a dict nor a list!")
            policies[leader_policy_id].set_weights(cur_weights)

    return RandomisePolicy0
