import copy
import os
import pickle
import re
import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import gym
import numpy as np
from ray.rllib.algorithms.a2c.a2c import A2C
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.pg.pg import PG
from ray.rllib.algorithms.sac.sac import SAC
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.registry import register_env as ray_register_env

from stackerlberg.core.envs import MultiAgentWrapper
from stackerlberg.train.make_env import make_repeated_matrix_observed_queries_env
from stackerlberg.trainers.callbacks import (
    DeleteHiddenQueriesCallback,
    DeleteHiddenQueriesPrePostprocessCallback,
    PolicyIntoEnv,
)
from stackerlberg.trainers.utils import randomise_leader_policy_each_episode
from stackerlberg.utils.utils import update_recursively

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker


def stackerlberg_trainable_es(config):
    """Ugly hack to make ES work with pre-trained follower policy."""

    # Some global configuration options:
    global_config = {
        "min_train_timesteps_per_iteration": 0,
        "min_time_s_per_iteration": 0,
        "min_sample_timesteps_per_iteration": 1,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 10,
        "train_batch_size": 10,
        "framework": "torch",
        "seed": np.random.randint(0, 2**32 - 1),
        "gamma": 1.0,
        "callbacks": DeleteHiddenQueriesCallback,
    }

    # Dict of callbacks:
    callbacks = {
        "config": [],
        "pre-follower": [],
        "post-follower": [],
        "pre-leader": [],
        "post-leader": [],
        "pre-outerloop": [],
        "post-outerloop": [],
        "pre-posttrain": [],
        "post-posttrain": [],
        "pre-pretrain": [],
        "post-pretrain": [],
    }
    callbacks.update(config.get("callbacks", {}))

    # Setting the seed from trial index, if using ray tune
    if "__trial_index__" in config:
        global_config["seed"] = config.pop("__trial_index__")

    # Get trainer and policy classes
    leader_trainer_cls = config.get("leader_algorithm", PG)
    follower_trainer_cls = config.get("follower_algorithm", PG)

    leader_policy_cls = config.get("leader_policy_class", None)
    follower_policy_cls = config.get("follower_policy_class", None)

    leader_agent_id = config.get("leader_agent_id", "agent_0")
    follower_agent_ids = config.get("follower_agent_ids", ["agent_1"])

    # Create configs:
    # We start with a global config, defined in this trainable, with some sensible defaults.
    leader_config = copy.deepcopy(global_config)
    if leader_trainer_cls == PG:
        leader_config["_disable_preprocessor_api"] = False
    follower_config = copy.deepcopy(global_config)
    if follower_trainer_cls == PG:
        follower_config["_disable_preprocessor_api"] = False

    # Then a common multiagent config for both
    multiagent_config = {
        "policies": {
            follower_agent_id: PolicySpec(follower_policy_cls, None, None, config.get("follower_policy_config", {}))
            for follower_agent_id in follower_agent_ids
        }
        | {leader_agent_id: PolicySpec(leader_policy_cls, None, None, config.get("leader_policy_config", {}))},
        "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
    }

    # Set this in both configs as a deepcopy.
    leader_config["multiagent"] = copy.deepcopy(multiagent_config)
    follower_config["multiagent"] = copy.deepcopy(multiagent_config)
    if config.get("_debug_dont_train_leader", False):
        # Debug config where we don't train the leader, useful only for hyperopt of the follower oracle.
        leader_config["multiagent"]["policies_to_train"] = []
    else:
        leader_config["multiagent"]["policies_to_train"] = [leader_agent_id]
    follower_config["multiagent"]["policies_to_train"] = follower_agent_ids

    # The we update the both configs first with the common config (passed in through the config dict),
    # and then the leader and follower specific configs
    leader_config.update(config.get("common_config", {}))
    follower_config.update(config.get("common_config", {}))
    leader_config.update(config.get("leader_config", {}))
    follower_config.update(config.get("follower_config", {}))

    # If the follower does Q-learning, we want to deactivate exploration during leader learning:
    if follower_trainer_cls in [DQN, SimpleQ] and config.get("deterministic_follower", True):
        for follower in follower_agent_ids:
            leader_config["multiagent"]["policies"][follower].config["explore"] = False
    # Vice versa for the leader during follower training:
    if leader_trainer_cls in [DQN, SimpleQ] and config.get("deterministic_leader", True):
        follower_config["multiagent"]["policies"][leader_agent_id].config["explore"] = False
    # Also if we override this in the config.
    # TODO double check this is correct. Intended behaviour is that for Q-learning, we default to no-exploration
    # while training other agent, but can override this in the config to have exploration ("deterministic_follower": False).
    # Default is deterministic = True for Q-learning, and deterministic = False for other algorithms.
    if config.get("deterministic_follower", False):
        for follower in follower_agent_ids:
            leader_config["multiagent"]["policies"][follower].config["explore"] = False
    # Vice versa for the leader during follower training:
    if config.get("deterministic_leader", False):
        follower_config["multiagent"]["policies"][leader_agent_id].config["explore"] = False

    # Create a config for post-training, before we potentially enable leader randomization!
    posttrain_config = copy.deepcopy(follower_config)
    pretrain_config = copy.deepcopy(follower_config)

    # ES specific
    leader_config.pop("multiagent", None)

    # Seed unique seeds for each trainer
    leader_config["seed"] += 1000
    follower_config["seed"] += 100000
    pretrain_config["seed"] += 10
    posttrain_config["seed"] += 100

    # Randomize leader during follower pre-training?
    if config.get("randomize_leader", False):
        if "callbacks" in pretrain_config:
            pretrain_config["callbacks"] = MultiCallbacks(
                [pretrain_config["callbacks"], randomise_leader_policy_each_episode(leader_agent_id)]
            )
        else:
            pretrain_config["callbacks"] = randomise_leader_policy_each_episode(leader_agent_id)

    for callback in callbacks["config"]:
        callback(
            leader_config=leader_config,
            follower_config=follower_config,
            pretrain_config=pretrain_config,
            posttrain_config=posttrain_config,
        )

    class WrapFollower(MultiAgentWrapper):
        def __init__(self, env, filename: str, **kwargs) -> None:
            super().__init__(env, **kwargs)
            with open(filename, "rb") as f:
                loaded_weights = pickle.load(f)
            my_config = copy.deepcopy(pretrain_config)
            self.trainer = follower_trainer_cls(config=my_config)
            self.trainer.set_weights(loaded_weights)
            self._agent_ids = {"agent_0"}
            self.action_space.spaces.pop("agent_1")
            self.observation_space.spaces.pop("agent_1")

        def reset(self):
            obs = self.env.reset()
            self.act_next_step = "agent_1" in obs
            if self.act_next_step:
                self._last_obs = obs["agent_1"]
                obs.pop("agent_1", None)
            return obs

        def step(self, action_dict):
            if self.act_next_step:
                action_dict["agent_1"] = self.trainer.compute_action(observation=self._last_obs)
            obs, rew, done, info = self.env.step(action_dict)
            self.act_next_step = "agent_1" in obs
            if self.act_next_step:
                self._last_obs = obs["agent_1"]
                obs.pop("agent_1", None)
            rew.pop("agent_1", None)
            done.pop("agent_1", None)
            return obs, rew, done, info

    class WrapFollowerGym(gym.Env):
        def __init__(self, env, filename: str, **kwargs) -> None:
            super().__init__()
            self.env = env
            with open(filename, "rb") as f:
                loaded_weights = pickle.load(f)
            my_config = copy.deepcopy(pretrain_config)
            self.trainer = follower_trainer_cls(config=my_config)
            self.trainer.set_weights(loaded_weights)
            self.action_space = self.env.action_space.spaces["agent_0"]
            self.observation_space = self.env.observation_space.spaces["agent_0"]

        def reset(self):
            obs = self.env.reset()
            self.act_next_step = "agent_1" in obs
            if self.act_next_step:
                self._last_obs = obs["agent_1"]
                obs.pop("agent_1", None)
            return obs["agent_0"]

        def step(self, ac):
            action_dict = {"agent_0": ac}
            if self.act_next_step:
                action_dict["agent_1"] = self.trainer.compute_action(observation=self._last_obs, policy_id="agent_1")
            obs, rew, done, info = self.env.step(action_dict)
            self.act_next_step = "agent_1" in obs
            if self.act_next_step:
                self._last_obs = obs["agent_1"]
                obs.pop("agent_1", None)
            return obs["agent_0"], rew.get("agent_0", 0), done.get("agent_0", False) or done.get("__all__", False), info

    def wrapped_env(env_config):
        env = make_repeated_matrix_observed_queries_env(**env_config)
        s = config.get("seed", 1) if 1 <= config.get("seed", 1) <= 5 else 1
        return WrapFollowerGym(
            env,
            os.path.dirname(os.path.realpath(__file__)) + f"/checkpoints_for_es/pretrain_checkpoint_prisoners_dilemma_{s}.pkl",
        )

    ray_register_env("wrapped_env", wrapped_env)
    leader_config["env"] = "wrapped_env"

    # Making a big try-finally block here to ensure that we always close the leader and follower trainers
    try:
        # Create trainers
        leader_trainer = leader_trainer_cls(config=leader_config)
        follower_trainer = follower_trainer_cls(config=follower_config)
        pre_trainer = follower_trainer_cls(config=pretrain_config)
        post_trainer = follower_trainer_cls(config=posttrain_config)

        # Start with empty results.
        results = {}

        # Load weights from pre-training phase
        if "pretrain_load_checkpoint" in config:
            with open(config["pretrain_load_checkpoint"], "rb") as f:
                loaded_weights = pickle.load(f)
                # leader_trainer.set_weights(loaded_weights)
                # follower_trainer.set_weights(loaded_weights)
                # pre_trainer.set_weights(loaded_weights)
                # post_trainer.set_weights(loaded_weights)

        # Optionally, do a pre-training phase to check we are actually in equilibrium.
        # Train follower
        # for pre_training_iteration in range(config.get("pre_training_iterations", 0)):
        #     # Callback
        #     for callback in callbacks["pre-pretrain"]:
        #         callback(
        #             leader_trainer=leader_trainer,
        #             follower_trainer=follower_trainer,
        #             pre_training_iteration=pre_training_iteration,
        #             pre_trainer=pre_trainer,
        #         )
        #     pretraining_results = pre_trainer.train()
        #     pretraining_results.pop("config", None)
        #     results["pretraining_results"] = pretraining_results
        #     results["phase"] = 0

        #     # Callback
        #     for callback in callbacks["post-pretrain"]:
        #         callback(
        #             leader_trainer=leader_trainer,
        #             follower_trainer=follower_trainer,
        #             pre_training_iteration=pre_training_iteration,
        #             results=results,
        #             pre_trainer=pre_trainer,
        #         )

        #     follower_trainer.set_weights(pre_trainer.get_weights(follower_agent_ids))
        #     leader_trainer.set_weights(pre_trainer.get_weights(follower_agent_ids))
        #     if config.get("log_weights", False):
        #         results["pre_weights"] = pre_trainer.get_weights()
        #     yield results

        #     if config.get("pre_training_stop_on_optimal", False) and results.get("follower_best_responds", {}).get("sum", -1) == 0:
        #         print("Stopping pre-training because optimal policy was found.")
        #         break

        # Save weights from pre-training phase
        if "pretrain_save_checkpoint" in config and config.get("pre_training_iterations", 0) > 0:
            if config["pretrain_save_checkpoint"] == "auto":
                config[
                    "pretrain_save_checkpoint"
                ] = f"pretrain_checkpoint_{config['common_config']['env_config']['matrix_name']}_{config['seed']}.pkl"
            with open(config["pretrain_save_checkpoint"], "wb") as f:
                pickle.dump(pre_trainer.get_weights(follower_agent_ids), f)

        # -------- Outer Loop --------
        for outer_loop_iteration in range(config.get("outer_iterations", 1)):
            # Callback
            for callback in callbacks["pre-outerloop"]:
                callback(
                    leader_trainer=leader_trainer,
                    follower_trainer=follower_trainer,
                    outer_loop_iteration=outer_loop_iteration,
                )
            # Train follower best-response:

            # First, optionally, we randomise the follower weights at the start of each iteration
            if config.get("randomize_follower", False):
                # OK, we're doing this somewhat drastically: Recreate a new follower trainer - this makes sure we reset replay buffer etc too.
                follower_trainer.stop()
                follower_config["seed"] += 1
                follower_trainer = follower_trainer_cls(config=follower_config)

            # Copy leader weights into the follower trainer
            # follower_trainer.set_weights(leader_trainer.get_weights(leader_agent_id))
            # -------- Follower Loop --------
            # Train followers
            # for follower_inner_loop_iteration in range(config.get("inner_iterations_follower", 1)):
            #     # Callback
            #     for callback in callbacks["pre-follower"]:
            #         callback(
            #             leader_trainer=leader_trainer,
            #             follower_trainer=follower_trainer,
            #             outer_loop_iteration=outer_loop_iteration,
            #             follower_inner_loop_iteration=follower_inner_loop_iteration,
            #         )
            #     follower_results = follower_trainer.train()
            #     follower_results.pop("config", None)
            #     results["follower_results"] = follower_results
            #     results["phase"] = 1

            #     # Callback
            #     for callback in callbacks["post-follower"]:
            #         callback(
            #             leader_trainer=leader_trainer,
            #             follower_trainer=follower_trainer,
            #             outer_loop_iteration=outer_loop_iteration,
            #             follower_inner_loop_iteration=follower_inner_loop_iteration,
            #             results=results,
            #         )
            #     if config.get("log_weights", False):
            #         results["follower_weights"] = follower_trainer.get_weights()
            #     yield results
            # -------- Follower Loop Done --------
            # Copy follower weights into the leader trainer
            # leader_trainer.set_weights(follower_trainer.get_weights(follower_agent_ids))
            # -------- Leader Loop --------
            # Train the leader
            for leader_inner_loop_iteration in range(config.get("inner_iterations_leader", 1)):
                # Callback
                for callback in callbacks["pre-leader"]:
                    callback(
                        leader_trainer=leader_trainer,
                        follower_trainer=follower_trainer,
                        outer_loop_iteration=outer_loop_iteration,
                        leader_inner_loop_iteration=leader_inner_loop_iteration,
                    )
                leader_results = leader_trainer.train()
                leader_results.pop("config", None)
                results["leader_results"] = leader_results
                results["phase"] = 1
                # Callback
                for callback in callbacks["post-leader"]:
                    callback(
                        leader_trainer=leader_trainer,
                        follower_trainer=follower_trainer,
                        outer_loop_iteration=outer_loop_iteration,
                        leader_inner_loop_iteration=leader_inner_loop_iteration,
                        results=results,
                    )
                if config.get("log_weights", False):
                    results["leader_weights"] = pre_trainer.get_weights()
                yield results
            # -------- Leader Loop Done --------
            # Callback
            for callback in callbacks["post-outerloop"]:
                callback(
                    leader_trainer=leader_trainer,
                    follower_trainer=follower_trainer,
                    outer_loop_iteration=outer_loop_iteration,
                    results=results,
                )
        # -------- Outer Loop Done --------

        # Optionally, do a post-training phase to check we are actually in equilibrium.
        # # Copy leader weights into follower trainer
        # post_trainer.set_weights(leader_trainer.get_weights())
        # # Train follower
        # for post_training_iteration in range(config.get("post_training_iterations", 0)):
        #     # Callback
        #     for callback in callbacks["pre-posttrain"]:
        #         callback(
        #             leader_trainer=leader_trainer,
        #             follower_trainer=follower_trainer,
        #             outer_loop_iteration=outer_loop_iteration,
        #             post_training_iteration=post_training_iteration,
        #             post_trainer=post_trainer,
        #         )
        #     posttraining_results = post_trainer.train()
        #     posttraining_results.pop("config", None)
        #     results["posttraining_results"] = posttraining_results
        #     results["phase"] = 2

        #     # We also evaluate the leader in its own trainer, to keep everythign the same there except follower weights.
        #     # Also useful so we always return results["evaluation"]. This is useful for ray tune.
        #     leader_trainer.set_weights(post_trainer.get_weights(follower_agent_ids))
        #     results["posttraining_results"]["evaluation"] = leader_trainer.evaluate()["evaluation"]
        #     # Callback
        #     for callback in callbacks["post-posttrain"]:
        #         callback(
        #             leader_trainer=leader_trainer,
        #             follower_trainer=follower_trainer,
        #             outer_loop_iteration=outer_loop_iteration,
        #             post_training_iteration=post_training_iteration,
        #             results=results,
        #             post_trainer=post_trainer,
        #         )
        #     if config.get("log_weights", False):
        #         results["post_weights"] = pre_trainer.get_weights()
        #     yield results

        print("Finished training.")

    finally:
        pre_trainer.stop()
        leader_trainer.stop()
        follower_trainer.stop()
        post_trainer.stop()
