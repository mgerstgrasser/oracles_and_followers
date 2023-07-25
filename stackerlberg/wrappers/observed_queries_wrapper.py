from collections import OrderedDict
from typing import Literal, Optional

import numpy as np
from gym import spaces

from stackerlberg.core.envs import (
    MultiAgentEnv,
    MultiAgentWrapper,
    ThreadedMultiAgentWrapper,
)


class ObservedQueriesWrapper(ThreadedMultiAgentWrapper):
    """Sends specified queries to leader policy at the beginning of each episode, and makes result available to followers.
    In theory, with sufficient queries and combined with curriculum learning, this should allow followers to best-respond instantly."""

    def __init__(
        self,
        env: MultiAgentEnv,
        leader_agent_id: str,
        queries: dict,
        n_samples: int = 1,
        samples_summarize: Literal["mean", "distribution", "list"] = "mean",
        tell_leader: bool = False,
        tell_leader_mock: bool = False,
        hidden_queries: bool = False,
        indentifier: Optional[str] = None,
    ):
        """Creates a new ObservedQueriesWrapper.
        Args:
            - env: MultiAgentEnv to wrap.
            - leader_agent_id: ID of the leader agent.
            - queries: Dictionary of queries to send to the leader.
            - n_samples: Number of samples to take for each query.
            - samples_summarize: Whether to summarize the samples as a distribution. If "list" (default), give list of responses.
                If "mean", summarize as scalar mean.
                If "distribution", give entire empirical distribution to followers.
            - tell_leader: Whether to tell the leader that the observation is a query, using an extra Discrete(2) input.
            - tell_leader_mock: If true, we still add a Discrete(2) but always set it to 0. Useful for pre-training.
            - hidden_queries: Whether to hide the queries from the leader, using a "hidden" key in the infos dict.
        """
        super().__init__(env, indentifier)
        self.leader = leader_agent_id
        self.queries = queries
        self.n_samples = n_samples
        self.samples_summarize = samples_summarize
        self.followers = self._agent_ids - {self.leader}
        self.tell_leader = tell_leader
        self.tell_leader_mock = tell_leader_mock
        self.hidden_queries = hidden_queries
        if not self.tell_leader:
            self.observation_space = {self.leader: env.observation_space[self.leader]}
        else:
            self.observation_space = {
                self.leader: spaces.Dict(OrderedDict(is_query=spaces.Discrete(2), original_space=env.observation_space[self.leader]))
            }
        # For each query, add the result (i.e. leader action) to each followers observation space
        for follower in self.followers:
            self.observation_space[follower] = OrderedDict({"original_space": env.observation_space[follower]})
            for q in self.queries:
                if self.samples_summarize == "list":
                    for i in range(self.n_samples):
                        self.observation_space[follower][f"{q}_{i}"] = env.action_space[self.leader]
                elif self.samples_summarize == "mean":
                    if isinstance(env.action_space[self.leader], spaces.Discrete):
                        self.observation_space[follower][f"{q}_0"] = spaces.Box(low=0, high=env.action_space[self.leader].n, shape=(1,))
                    elif isinstance(env.action_space[self.leader], spaces.Box) and env.action_space[self.leader].shape == (1,):
                        self.observation_space[follower][f"{q}_0"] = spaces.Box(
                            low=env.action_space[self.leader].low, high=env.action_space[self.leader].high, shape=(1,)
                        )
                    else:
                        raise NotImplementedError(f"Unsupported action space {env.action_space[self.leader]}")
                elif self.samples_summarize == "distribution":
                    raise NotImplementedError("Distribution summarization not implemented yet")
                else:
                    raise NotImplementedError(f"Unknown summarize type {self.samples_summarize}")
                    # self.observation_space[follower][q] = env.action_space[self.leader]
            self.observation_space[follower] = spaces.Dict(self.observation_space[follower])
        self.observation_space = spaces.Dict(self.observation_space)

    def run(self):
        """Runs one episode"""
        query_results = {}
        # Query leader
        for q in self.queries:
            these_results = []
            for _ in range(self.n_samples):
                if not self.tell_leader:
                    these_results.append(self._thr_get_actions({self.leader: self.queries[q]}, hidden=self.hidden_queries)[self.leader])
                elif self.tell_leader_mock:
                    these_results.append(
                        self._thr_get_actions(
                            {
                                self.leader: OrderedDict(is_query=0, original_space=self.queries[q]),
                            },
                            hidden=self.hidden_queries,
                        )[self.leader]
                    )
                else:
                    these_results.append(
                        self._thr_get_actions(
                            {
                                self.leader: OrderedDict(is_query=1, original_space=self.queries[q]),
                            },
                            hidden=self.hidden_queries,
                        )[self.leader]
                    )
            if self.samples_summarize == "list":
                for i in range(self.n_samples):
                    query_results[f"{q}_{i}"] = these_results[i]
            elif self.samples_summarize == "mean":
                query_results[f"{q}_0"] = np.array([np.mean(these_results)], dtype=np.float32)
            elif self.samples_summarize == "distribution":
                raise NotImplementedError("Distribution summarization not implemented yet")
            else:
                raise NotImplementedError(f"Unknown summarize type {self.samples_summarize}")
        # Now real episode begins
        dones = {}
        obs = self.env.reset()
        while not ("__all__" in dones and dones["__all__"] is True):
            for follower in obs.keys() & self.followers:
                obs[follower] = OrderedDict({"original_space": obs[follower]})
                for q in query_results:
                    obs[follower][q] = query_results[q]
            if self.leader in obs and self.tell_leader:
                obs[self.leader] = OrderedDict(is_query=0, original_space=obs[self.leader])
            actions = self._thr_get_actions(obs)
            obs, rewards, dones, infos = self.env.step(actions)
            self._thr_log_rewards(rewards)
            self._thr_log_info(infos)
            self._thr_set_dones(dones)
        # Episode is done, send one final observation to agents.
        for follower in obs.keys() & self.followers:
            obs[follower] = OrderedDict({"original_space": obs[follower]})
            for q in query_results:
                obs[follower][q] = query_results[q]
        if self.leader in obs and self.tell_leader:
            obs[self.leader] = OrderedDict(is_query=0, original_space=obs[self.leader])
        self._thr_end_episode(obs)
