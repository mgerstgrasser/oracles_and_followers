import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from stackerlberg.core.envs import MultiAgentWrapper

named_matrix_games = {
    "prisoners_dilemma": [
        [[3, 3], [1, 4]],
        [[4, 1], [2, 2]],
    ],
    "stag_hunt": [
        [[4, 4], [1, 3]],
        [[3, 1], [2, 2]],
    ],
    "assurance": [
        [[4, 4], [1, 2]],
        [[2, 1], [3, 3]],
    ],
    "coordination": [
        [[4, 4], [2, 1]],
        [[1, 2], [3, 3]],
    ],
    "mixedharmony": [
        [[4, 4], [3, 1]],
        [[1, 3], [2, 2]],
    ],
    "harmony": [
        [[4, 4], [3, 2]],
        [[2, 3], [1, 1]],
    ],
    "noconflict": [
        [[4, 4], [2, 3]],
        [[3, 2], [1, 1]],
    ],
    "deadlock": [
        [[2, 2], [1, 4]],
        [[4, 1], [3, 3]],
    ],
    "prisoners_delight": [
        [[1, 1], [2, 4]],
        [[4, 2], [3, 3]],
    ],
    "hero": [
        [[1, 1], [3, 4]],
        [[4, 3], [2, 2]],
    ],
    "battle": [
        [[2, 2], [3, 4]],
        [[4, 3], [1, 1]],
    ],
    "chicken": [
        [[3, 3], [2, 4]],
        [[4, 2], [1, 1]],
    ],
}


class MatrixGameEnv(MultiAgentEnv):
    """A very basic marix game environment."""

    def __init__(
        self,
        matrix: np.ndarray or str = "prisoners_dilemma",
        episode_length: int = 1,
        memory: bool = False,
        small_memory: bool = False,
        reward_offset: float = -2.5,
        **kwargs,
    ):
        """Creates a simple matrix game.
        Arguments:

        - matrix: A 3D numpy array of shape (rows, cols, 2) containing the payoff (bi-)matrix. Alternatively, a string can be passed, identifying one of several canonical games.
        - episode_length: The length of an episode.
        - memory: If True, agents can see the previous action of both agents."""
        super().__init__()
        if isinstance(matrix, str):
            matrix = np.array(named_matrix_games[matrix])
        self.matrix = matrix
        self.num_agents = 2
        self._agent_ids = {"agent_0", "agent_1"}
        self.action_space = spaces.Dict(
            {
                "agent_0": spaces.Discrete(len(matrix)),
                "agent_1": spaces.Discrete(len(matrix[0])),
            }
        )
        self.memory = memory
        self.small_memory = small_memory
        if memory is False:
            self.observation_space = spaces.Dict(
                {
                    "agent_0": spaces.Discrete(1),
                    "agent_1": spaces.Discrete(1),
                }
            )
        else:
            if small_memory is False:
                self.observation_space = spaces.Dict(
                    {
                        "agent_0": spaces.Discrete(5),
                        "agent_1": spaces.Discrete(5),
                    }
                )
            else:
                self.observation_space = spaces.Dict(
                    {
                        "agent_0": spaces.Discrete(3),
                        "agent_1": spaces.Discrete(3),
                    }
                )
        self.episode_length = episode_length
        self.current_step = 0
        self.reward_offset = reward_offset

    def reset(self):
        self.current_step = 0
        return {"agent_0": 0, "agent_1": 0}

    def step(self, actions):
        self.current_step += 1
        rewards = {
            "agent_0": self.matrix[actions["agent_0"]][actions["agent_1"]][0] + self.reward_offset,
            "agent_1": self.matrix[actions["agent_0"]][actions["agent_1"]][1] + self.reward_offset,
        }
        if self.memory is False:
            obs = {"agent_0": 0, "agent_1": 0}
        else:
            if self.small_memory is False:
                obs = {
                    "agent_0": 1 + actions["agent_0"] + 2 * actions["agent_1"],
                    "agent_1": 1 + actions["agent_0"] + 2 * actions["agent_1"],
                }
                # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
                # 1, 2: agent 1 action 0, 3, 4 action 1
                # 1, 3: agent 0 action 0, 2, 4 action 1
            else:
                obs = {"agent_0": 1 + actions["agent_1"], "agent_1": 1 + actions["agent_0"]}
                # 0: first step
                # 1: other agent cooperated
                # 2: other agent defected
        return obs, rewards, {"__all__": True if self.current_step >= self.episode_length else False}, {}


class StochasticRewardWrapper(MultiAgentWrapper):
    """Makes reward stochastich and sparser, but with same expectation."""

    def __init__(self, env, prob: float = 1, scale: float = 1, agent: str = "agent_1", deterministic: bool = False, **kwargs):
        super().__init__(env, **kwargs)
        self.scale = scale
        self.prob = prob
        self.agent = agent
        self.deterministic = deterministic

    def reset(self):
        self._step_counter = 0
        return self.env.reset()

    def step(self, actions):
        self._step_counter += 1
        obs, rewards, dones, infos = self.env.step(actions)
        if self.agent in rewards:
            if self.deterministic:
                rewards[self.agent] = rewards[self.agent] * self.scale if self._step_counter >= self.prob else 0
            else:
                rewards[self.agent] *= self.scale * np.random.binomial(1, 1 / self.prob)
        return obs, rewards, dones, infos
