import copy
from math import isnan
from typing import Optional

import numpy as np
from gym import spaces

from stackerlberg.core.envs import MultiAgentEnv, MultiAgentWrapper


class DictToDiscreteObsWrapper(MultiAgentWrapper):
    """This wrapper converts a Dict(Discrete) observation space into a Discrete observation space."""

    def __init__(self, env: MultiAgentEnv, agent_id: str = "agent_1", indentifier: Optional[str] = None):
        super().__init__(env, indentifier)
        self.agent_id = agent_id
        self.observation_space = copy.deepcopy(env.observation_space)
        assert isinstance(env.observation_space[agent_id], spaces.Dict), "DictToDiscreteObsWrapper only works with Dict observation spaces"
        obs_space_size = 1
        for obs_type, space in env.observation_space[agent_id].spaces.items():
            assert isinstance(space, spaces.Discrete), "DictToDiscreteObsWrapper only works with Dict(Discrete) observation spaces"
            dim_size = space.n
            obs_space_size *= dim_size
        self.observation_space[agent_id] = spaces.Discrete(obs_space_size)

    def encode_obs(self, obs):
        discrete_obs = 0
        for obs_type, obs_value in obs.items():
            discrete_obs *= self.env.observation_space[self.agent_id].spaces[obs_type].n
            discrete_obs += obs_value
        return discrete_obs

    def reset(self):
        observation = self.env.reset()
        if self.agent_id in observation:
            agent_obs = observation[self.agent_id]  # this is a dict.
            observation[self.agent_id] = self.encode_obs(agent_obs)
        return observation

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if self.agent_id in observation:
            agent_obs = observation[self.agent_id]  # this is a dict.
            observation[self.agent_id] = self.encode_obs(agent_obs)
        return observation, reward, done, info
