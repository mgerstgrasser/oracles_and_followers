from math import isnan
from typing import Optional

import numpy as np
from gym import spaces

from stackerlberg.core.envs import MultiAgentEnv, MultiAgentWrapper


class ActionToDistWrapper(MultiAgentWrapper):
    """This wrapper takes discrete action spaces and converts them to action distribution spaces,
    e.g. a spaces.Discrete(3) will transform into a spaces.Box()"""

    def __init__(self, env: MultiAgentEnv, indentifier: Optional[str] = None):
        super().__init__(env, indentifier)
        self.action_space = {}
        for agent in self._agent_ids:
            assert isinstance(self.env.action_space[agent], spaces.Discrete), "ActionToDistWrapper only works with discrete action spaces"
            self.action_space[agent] = spaces.Box(low=0, high=1, shape=(self.env.action_space[agent].n - 1,))
        self.action_space = spaces.Dict(self.action_space)

    def step(self, actions):
        for agent in actions:
            actions[agent] = np.append(actions[agent], 1 - np.sum(actions[agent]))
            if np.sum(actions[agent]) != 0 and not isnan(np.sum(actions[agent])):
                p = actions[agent] / np.sum(actions[agent])
            else:
                p = None
            actions[agent] = np.random.choice(self.env.action_space[agent].n, p=p)
        return self.env.step(actions)
