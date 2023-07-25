"""Script to calculate performance targets for pretraining."""
import itertools

import numpy as np

from stackerlberg.envs.matrix_game import MatrixGameEnv


def eval_matrix_game_deterministic(env, policy0, policy1):
    obs = env.reset()
    rewards = {"agent_0": 0, "agent_1": 0}
    dones = {"__all__": False}
    while not ("__all__" in dones and dones["__all__"]):
        actions = {"agent_0": policy0[obs["agent_0"]], "agent_1": policy1[obs["agent_1"]]}
        obs, reward, dones, info = env.step(actions)
        for agent in reward:
            rewards[agent] += reward[agent]
    return rewards


def calculate_pretrain_perf_targets(env):
    """Calculate performance targets for pretraining."""

    ndim = env.observation_space["agent_0"].n
    assert env.action_space["agent_0"].n == 2
    assert env.action_space["agent_1"].n == 2

    all_policies = list(itertools.product(*[(0, 1)] * ndim))
    performance_targets = {}
    best_followers = {}
    for leader_policy in all_policies:
        max_perf_this_leader = None
        best_follower_this_leader = None
        for follower_policy in all_policies:
            perf_this_follower = eval_matrix_game_deterministic(env, leader_policy, follower_policy)
            if max_perf_this_leader is None or perf_this_follower["agent_1"] > max_perf_this_leader:
                max_perf_this_leader = perf_this_follower["agent_1"]
                best_follower_this_leader = follower_policy
        performance_targets[leader_policy] = max_perf_this_leader
        best_followers[leader_policy] = best_follower_this_leader
    # print(performance_targets)
    return performance_targets, best_followers


if __name__ == "__main__":
    matrix = "prisoners_dilemma_2"
    calculate_pretrain_perf_targets(matrix, memory=True, small_memory=False, episode_length=10)
