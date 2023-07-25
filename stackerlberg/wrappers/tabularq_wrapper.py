from collections import OrderedDict, defaultdict
from typing import Literal, Optional

import numpy as np
from gym import spaces

from stackerlberg.core.envs import (
    MultiAgentEnv,
    MultiAgentWrapper,
    ThreadedMultiAgentWrapper,
)


class TabularQWrapper(ThreadedMultiAgentWrapper):
    """This wrapper takes the follower agent and trains it using tabular Q-learning.
    For each episode it presents to the outside world, it will run an entire Q-learning training.
    When the outside-world-episode is done, it will by default reset the follower Q-Table.
    Only the leader agent will be shown to the outside world, making this a single-agent environment.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        leader_agent_id: str,
        follower_agent_id: str,
        n_q_episodes: int = 10,
        reset_between_episodes: bool = True,
        epsilon: float = 1.0,
        alpha: float = 0.1,
        gamma: float = 1.0,
        param_noise: bool = False,
        tell_leader: bool = False,
        tell_leader_mock: bool = False,
        hidden_queries: bool = False,
        leader_reward_during_q: bool = False,
        q_init_zero: bool = False,
        indentifier: Optional[str] = None,
    ):
        """Creates a new TabularQWrapper.
        Args:
            - env: MultiAgentEnv to wrap.
            - leader_agent_id: ID of the leader agent.
            - follower_agent_id: ID of the follower agent.
            - n_q_episodes: Number of episodes to run for q-learning.
            - epsilon, alpha, gamma: usual q-learning hyperparameters.
            - reset_between_episodes: Whether to reset the follower q-table between episodes.
            - param_noise: Use parameter-noise for the follower Q-learning.
            - tell_leader: (debug only) Whether to tell the leader that the observation is a query, using an extra Discrete(2) input.
            - tell_leader_mock: (debug only) If true, we still add a Discrete(2) but always set it to 0. Useful for pre-training.
            - hidden_queries: (debug only) Whether to hide the queries from the leader, using a "hidden" key in the infos dict.
            - leader_reward_during_q: (debug only) Whether to give the leader a reward during the q-learning phase.
            - q_init_zero: (debug only) Whether to initialize the Q-table to all zeros.
            - indentifier: (debug only) Identifier for the wrapper, not currently used.
        """
        super().__init__(env, indentifier)
        self.leader = leader_agent_id
        self.follower = follower_agent_id
        self.n_q_episodes = n_q_episodes
        self.reset_between_episodes = reset_between_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.tell_leader = tell_leader
        self.tell_leader_mock = tell_leader_mock
        self.hidden_queries = hidden_queries
        self.param_noise = param_noise
        self.leader_reward_during_q = leader_reward_during_q
        self.q_init_zero = q_init_zero
        if not self.tell_leader:
            self.observation_space = {self.leader: env.observation_space[self.leader]}
        else:
            self.observation_space = {
                self.leader: spaces.Dict(OrderedDict(is_query=spaces.Discrete(2), original_space=env.observation_space[self.leader]))
            }
        # Remove follower from outside world.
        self.observation_space = spaces.Dict(self.observation_space)
        self.action_space = spaces.Dict({self.leader: env.action_space[self.leader]})
        self._agent_ids = {self.leader}
        # Q-Table
        if self.q_init_zero:
            self.q_table = defaultdict(lambda: np.zeros(env.action_space[self.follower].n))
        else:
            self.q_table = defaultdict(lambda: np.random.normal(0.0, 0.01, env.action_space[self.follower].n))

    def run(self):
        """Runs one episode"""
        # Run Q-learning.
        if self.reset_between_episodes:
            if self.q_init_zero:
                self.q_table = defaultdict(lambda: np.zeros(self.env.action_space[self.follower].n))
            else:
                self.q_table = defaultdict(lambda: np.random.normal(0.0, 0.01, self.env.action_space[self.follower].n))
        # for debugging:
        leader_actions_this_episode = []
        leader_reward_this_episode = 0
        follower_actions_this_episode = []
        for n_episode in range(self.n_q_episodes):
            obs = self.env.reset()
            dones = {}
            # Let's try parameter noise?
            self.q_table_noise = defaultdict(lambda: np.random.normal(0.0, self.epsilon, self.env.action_space[self.follower].n))
            while not ("__all__" in dones and dones["__all__"]):
                # rollout episode
                # Get leader action
                if not self.tell_leader:
                    leader_action = self._thr_get_actions({self.leader: obs[self.leader]}, hidden=self.hidden_queries)[self.leader]
                elif self.tell_leader_mock:
                    leader_action = self._thr_get_actions(
                        {self.leader: OrderedDict(is_query=0, original_space=obs[self.leader])},
                        hidden=self.hidden_queries,
                    )[self.leader]

                else:
                    leader_action = self._thr_get_actions(
                        {self.leader: OrderedDict(is_query=1, original_space=obs[self.leader])},
                        hidden=self.hidden_queries,
                    )[self.leader]

                if self.param_noise:
                    follower_action = (self.q_table[obs[self.leader]] + self.q_table_noise[obs[self.leader]]).argmax()
                else:
                    follower_action = (self.q_table[obs[self.leader]]).argmax()
                    if np.random.rand() < self.epsilon:
                        follower_action = self.env.action_space[self.follower].sample()
                prev_obs = obs[self.follower]

                # for debugging:
                leader_actions_this_episode.append(leader_action)
                follower_actions_this_episode.append(follower_action)

                obs, rewards, dones, infos = self.env.step({self.leader: leader_action, self.follower: follower_action})

                # for debugging:
                leader_reward_this_episode += rewards[self.leader]

                if not ("__all__" in dones and dones["__all__"]):
                    self.q_table[prev_obs][follower_action] = self.q_table[prev_obs][follower_action] + self.alpha * (
                        rewards[self.follower]
                        + self.gamma * max(self.q_table[obs[self.follower]])
                        - self.q_table[prev_obs][follower_action]
                    )
                else:
                    # At episode end, we don't take reward from next state!
                    self.q_table[prev_obs][follower_action] = self.q_table[prev_obs][follower_action] + self.alpha * (
                        rewards[self.follower] - self.q_table[prev_obs][follower_action]
                    )

                # Optionally give reward to leader during q-learning.
                if self.leader_reward_during_q:
                    rewards.pop(self.follower, None)
                    self._thr_log_rewards(rewards)

        # Now real episode begins
        dones = {}
        obs = self.env.reset()
        while not ("__all__" in dones and dones["__all__"] is True):
            # If we tell the leader it's not a query
            if self.leader in obs and self.tell_leader:
                obs[self.leader] = OrderedDict(is_query=0, original_space=obs[self.leader])
            # Get leader action
            leader_action = self._thr_get_actions({self.leader: obs[self.leader]})[self.leader]
            # Get follower action, no epsilon-greedy here
            follower_action = self.q_table[obs[self.follower]].argmax()
            # TODO do we still do epsilon-greedy here?
            # if np.random.rand() < self.epsilon:
            #     follower_action = self.env.action_space[self.follower].sample()
            # Assemble action dict
            actions = {self.leader: leader_action, self.follower: follower_action}
            # step inner env
            obs, rewards, dones, infos = self.env.step(actions)
            # Remove follower from rewards, infos, dones.
            rewards.pop(self.follower, None)
            infos.pop(self.follower, None)
            dones.pop(self.follower, None)
            # Log rewards, infos, dones, for leader.
            self._thr_log_rewards(rewards)
            self._thr_log_info(infos)
            self._thr_set_dones(dones)
            leader_actions_this_episode.append(leader_action)
            follower_actions_this_episode.append(follower_action)
            leader_reward_this_episode += rewards[self.leader]
        # Episode is done, send one final observation to agents.
        if self.leader in obs and self.tell_leader:
            obs[self.leader] = OrderedDict(is_query=0, original_space=obs[self.leader])
        self._thr_end_episode({self.leader: obs[self.leader]})
