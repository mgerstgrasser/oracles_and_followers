import copy

import numpy as np

from stackerlberg.envs.matrix_game import MatrixGameEnv
from stackerlberg.train.experiments.calculate_pretrain_targets import (
    calculate_pretrain_perf_targets,
    eval_matrix_game_deterministic,
)
from stackerlberg.train.policies import IPDRandomEveryEpisodePolicy


def pretrain_against_random(pretrain_config={}, **kwargs):
    """Pretrain against a random agent."""
    pretrain_config["multiagent"]["policies"]["agent_0"].policy_class = IPDRandomEveryEpisodePolicy


def check_follower_best_responds_pretrain(pre_trainer=None, results={}, **kwargs):
    """Check if the follower is best responding to the leader's strategy."""
    # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
    # 1, 2: agent 1 action 0, 3, 4 action 1
    # 1, 3: agent 0 action 0, 2, 4 action 1
    results["follower_best_responds"] = {}
    if "discrete_obs" in pre_trainer.config["env_config"] and pre_trainer.config["env_config"]["discrete_obs"]:
        # The case where we multiply out the entire observation space
        obs = lambda o: o * 32
        results["follower_best_responds"]["allcoop"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 1, 3))) else 0
        )
        obs = lambda o: o * 32 + 31
        results["follower_best_responds"]["alldef"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 2, 4))) else 0
        )
        obs = lambda o: o * 32 + 3
        results["follower_best_responds"]["tftfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
        )
        obs = lambda o: o * 32 + 19
        results["follower_best_responds"]["tftunfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
        )
        results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())
    else:
        if "samples_summarize" in pre_trainer.config["env_config"] and pre_trainer.config["env_config"]["samples_summarize"] == "mean":
            O = np.array([0.0], dtype=np.float32)
            I = np.array([1.0], dtype=np.float32)
        else:
            O = 0
            I = 1
        obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": O, "q3_0": O, "q4_0": O, "original_space": o}
        results["follower_best_responds"]["allcoop"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 1, 3))) else 0
        )
        obs = lambda o: {"q0_0": I, "q1_0": I, "q2_0": I, "q3_0": I, "q4_0": I, "original_space": o}
        results["follower_best_responds"]["alldef"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 2, 4))) else 0
        )
        obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": O, "q3_0": I, "q4_0": I, "original_space": o}
        results["follower_best_responds"]["tftfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
        )
        obs = lambda o: {"q0_0": I, "q1_0": O, "q2_0": O, "q3_0": I, "q4_0": I, "original_space": o}
        results["follower_best_responds"]["tftunfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
        )
        results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())


def check_follower_best_responds_maintrain(follower_trainer=None, results={}, **kwargs):
    """Check if the follower is best responding to the leader's strategy."""
    results["follower_best_responds"] = {}
    if (
        "samples_summarize" in follower_trainer.config["env_config"]
        and follower_trainer.config["env_config"]["samples_summarize"] == "mean"
    ):
        O = np.array([0.0], dtype=np.float32)
        I = np.array([1.0], dtype=np.float32)
    else:
        O = 0
        I = 1
    obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": O, "q3_0": O, "q4_0": O, "original_space": o}
    results["follower_best_responds"]["allcoop"] = (
        1 if all((follower_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 1, 3))) else 0
    )
    obs = lambda o: {"q0_0": I, "q1_0": I, "q2_0": I, "q3_0": I, "q4_0": I, "original_space": o}
    results["follower_best_responds"]["alldef"] = (
        1 if all((follower_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 2, 4))) else 0
    )
    obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": O, "q3_0": I, "q4_0": I, "original_space": o}
    results["follower_best_responds"]["tftfriendly"] = (
        1 if all((follower_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
    )
    obs = lambda o: {"q0_0": I, "q1_0": O, "q2_0": O, "q3_0": I, "q4_0": I, "original_space": o}
    results["follower_best_responds"]["tftunfriendly"] = (
        1 if all((follower_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1, 3))) else 0
    )
    results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())


# prep = get_preprocessor(follower_trainer.workers._local_worker.env.observation_space["agent_1"])(follower_trainer.workers._local_worker.env.observation_space["agent_1"])
# prep.transform(obs)
# pol.model({"obs": [obs_transform]})
# ...see rllib querying action distributions


def smipd_check_follower_best_responds_pretrain(pre_trainer=None, results={}, **kwargs):
    """Check if the follower is best responding to the leader's strategy."""

    # 0: first step
    # 1: other agent cooperated
    # 2: other agent defected
    results["follower_best_responds"] = {}
    if "discrete_obs" in pre_trainer.config["env_config"] and pre_trainer.config["env_config"]["discrete_obs"]:
        # The case where we multiply out the entire observation space
        obs = lambda o: o * 8
        results["follower_best_responds"]["allcoop"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in [0, 1])) else 0
        )
        obs = lambda o: o * 8 + 7
        results["follower_best_responds"]["alldef"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in [0, 2])) else 0
        )
        obs = lambda o: o * 8 + 1
        results["follower_best_responds"]["tftfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1))) else 0
        )
        obs = lambda o: o * 8 + 5
        results["follower_best_responds"]["tftunfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1))) else 0
        )
        results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())
    else:
        if "samples_summarize" in pre_trainer.config["env_config"] and pre_trainer.config["env_config"]["samples_summarize"] == "mean":
            O = np.array([0.0], dtype=np.float32)
            I = np.array([1.0], dtype=np.float32)
        else:
            O = 0
            I = 1
        obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": O, "original_space": o}
        results["follower_best_responds"]["allcoop"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 1))) else 0
        )
        obs = lambda o: {"q0_0": I, "q1_0": I, "q2_0": I, "original_space": o}
        results["follower_best_responds"]["alldef"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 1 for o in (0, 2))) else 0
        )
        obs = lambda o: {"q0_0": O, "q1_0": O, "q2_0": I, "original_space": o}
        results["follower_best_responds"]["tftfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1))) else 0
        )
        obs = lambda o: {"q0_0": I, "q1_0": O, "q2_0": I, "original_space": o}
        results["follower_best_responds"]["tftunfriendly"] = (
            1 if all((pre_trainer.compute_single_action(obs(o), policy_id="agent_1", explore=False) == 0 for o in (0, 1))) else 0
        )
        results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())


def get_follower_policy(trainer, leader_policy):
    """Get the follower policy."""
    obs_size = 3 if "small_memory" in trainer.config["env_config"] and trainer.config["env_config"]["small_memory"] else 5
    if "discrete_obs" in trainer.config["env_config"] and trainer.config["env_config"]["discrete_obs"]:
        # The case where we multiply out the entire observation space
        # First we get the encoded leader policy
        leader_pol_encoded = sum([2**i * list(reversed(leader_policy))[i] for i in range(len(leader_policy))])
        # Then we get the follower action for this leader policy and each possible observation
        pol = (trainer.compute_single_action(8 * o + leader_pol_encoded, policy_id="agent_1", explore=False) for o in range(obs_size))
    else:
        if "samples_summarize" in trainer.config["env_config"] and trainer.config["env_config"]["samples_summarize"] == "mean":
            O = np.array([0.0], dtype=np.float32)
            I = np.array([1.0], dtype=np.float32)
        else:
            O = 0
            I = 1
        obs = {f"q{i}_0": O if leader_policy[i] else I for i in range(len(leader_policy))}
        pol = (trainer.compute_single_action({"original_space": o, **obs}, policy_id="agent_1", explore=False) for o in range(obs_size))
    return list(pol)


def smipd_check_follower_best_response(results={}, **kwargs):
    if "pre_trainer" in kwargs:
        trainer = kwargs["pre_trainer"]
    elif "follower_trainer" in kwargs:
        trainer = kwargs["follower_trainer"]
    elif "post_trainer" in kwargs:
        trainer = kwargs["post_trainer"]
    else:
        raise ValueError
    env_config = copy.deepcopy(trainer.config["env_config"])
    if "matrix_name" in env_config:
        env_config["matrix"] = env_config["matrix_name"]
    if "episode_length" not in env_config:
        env_config["episode_length"] = 10
    if "memory" not in env_config:
        env_config["memory"] = True
    env = MatrixGameEnv(**env_config)
    performance_targets, best_followers = calculate_pretrain_perf_targets(env)
    results["follower_best_responds"] = {}
    for leader_policy in performance_targets:
        follower_policy = get_follower_policy(trainer, leader_policy)
        performance_this_follower = eval_matrix_game_deterministic(env, leader_policy, follower_policy)["agent_1"]
        results["follower_best_responds"][f"leader_{str(leader_policy)}"] = performance_this_follower - performance_targets[leader_policy]
    results["follower_best_responds"]["sum"] = sum(results["follower_best_responds"].values())
    env.close()
