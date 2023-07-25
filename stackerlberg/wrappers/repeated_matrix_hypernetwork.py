from math import isnan
from typing import Literal, Optional

import numpy as np
from gym import spaces

from stackerlberg.core.envs import (
    MultiAgentEnv,
    MultiAgentWrapper,
    ThreadedMultiAgentWrapper,
)

# Not used in the O&F paper.


class RepeatedMatrixHypernetworkWrapper(ThreadedMultiAgentWrapper):
    """This wrapper takes a repeated matrix game, and converts it into a hypernetwork game for the leader.
    Given five states and 2 actions, the leader gets a 5-dim box, denoting the probability of it choosing action 0 in each of the states."""

    def __init__(
        self,
        env: MultiAgentEnv,
        leader_agent_id: str = "agent_0",
        discrete: bool = False,
        queries: bool = False,
        indentifier: Optional[str] = None,
    ):
        super().__init__(env, indentifier)
        self.leader_agent_id = leader_agent_id
        self.follower_agent_id = [agent for agent in env._agent_ids if agent != leader_agent_id][0] if len(env._agent_ids) > 1 else None
        self.discrete = discrete
        assert isinstance(self.observation_space[leader_agent_id], spaces.Discrete)
        assert isinstance(self.action_space[leader_agent_id], spaces.Discrete)
        assert self.action_space[leader_agent_id].n == 2
        if self.discrete:
            self.action_space[leader_agent_id] = spaces.MultiBinary(self.observation_space[leader_agent_id].n)
        else:
            self.action_space[leader_agent_id] = spaces.Box(low=0, high=1, shape=(self.observation_space[leader_agent_id].n,))
        self.observation_space[leader_agent_id] = spaces.Discrete(1)  # spaces.Box(low=0, high=1, shape=(1,))
        self.queries = queries
        if self.queries:
            self.observation_space[self.follower_agent_id] = spaces.Dict(
                {
                    "original_space": env.observation_space[self.follower_agent_id],
                    "queries": self.action_space[leader_agent_id],
                }
            )

    def run(self):
        """Runs one episode"""
        # leader_actions = self._thr_get_actions({self.leader_agent_id: np.array([1.0], dtype=np.float32)})[self.leader_agent_id]
        leader_actions = self._thr_get_actions({self.leader_agent_id: 0})[self.leader_agent_id]
        # Now real episode begins
        dones = {}
        obs = self.env.reset()
        while not ("__all__" in dones and dones["__all__"] is True):
            # Get follower actions from agents, if there are any followers to query.
            if len({agent: obs[agent] for agent in obs if agent != self.leader_agent_id}) != 0:
                if self.queries:
                    obs[self.follower_agent_id] = {
                        "original_space": obs[self.follower_agent_id],
                        "queries": leader_actions,
                    }
                actions = self._thr_get_actions({self.follower_agent_id: obs[self.follower_agent_id]})
            else:
                actions = {}
            # Get leader action from coin flip
            if self.discrete:
                leader_action = leader_actions[obs[self.leader_agent_id]]
            else:
                leader_action_prob = leader_actions[obs[self.leader_agent_id]]
                leader_action = np.random.choice([0, 1], p=[leader_action_prob, 1 - leader_action_prob])
            actions[self.leader_agent_id] = leader_action
            obs, rewards, dones, infos = self.env.step(actions)
            self._thr_log_rewards(rewards)
            self._thr_log_info(infos)
            self._thr_set_dones(dones)
        # Episode is done, send one final observation to leader:
        # obs[self.leader_agent_id] = np.array([1.0], dtype=np.float32)
        obs[self.leader_agent_id] = 0
        self._thr_end_episode(obs)
