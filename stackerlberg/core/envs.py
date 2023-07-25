import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from typing import List, Optional, Set, Tuple, Union

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class MultiAgentWrapper(MultiAgentEnv):
    """Base wrapper class of MultiAgentEnv"""

    def __init__(self, env: MultiAgentEnv, indentifier: Optional[str] = None):
        """Initializes a new MultiAgentWrapper

        Args:
            env (MultiAgentEnv): The environment to wrap.
            stats_descriptor (Optional[str]): The descriptor of wrapper's statistics.
        """
        # TODO figure out multiple inheritance here!
        # super().__init__()
        self.env = env
        self.indentifier = indentifier
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = self.env._agent_ids
        self._spaces_in_preferred_format = True

    def __getattr__(self, name: str):
        # If we try to access an attribute that doesn't exist in this wrapper, we look for it in the wrapped env.
        # But we need to check first that there is a wrapped env! Otherwise we call __getattr__ again on self.env,
        # resulting in infinite recursion.
        if "env" in dir(self):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        return self.env.step(actions)

    def reset(self) -> MultiAgentDict:
        return self.env.reset()

    @property
    def unwrapped(self) -> MultiAgentEnv:
        if "unwrapped" in dir(self.env):
            return self.env.unwrapped
        else:
            return self.env


class ThreadedMultiAgentEnv(MultiAgentEnv):
    """This is a base class for environments that maintain their own thread of control, similar to ExternalEnv.
    However, this is based off MultiAgentEnv, so the logic is step() and reset() rather than poll() and send_actions(),
    and it is all in one environment class. That means you can more easily wrap it in standard environment wrappers."""

    def __init__(self):
        """Initialize the environment.
        You can override this in subclasses, but you must call super().__init__()."""
        # super().__init__()
        MultiAgentEnv.__init__(self)
        self._spaces_in_preferred_format = True
        # Queues for sending actions and observations back and forth.
        # Set maxsize to 1, there should never be more than one query in-flight.
        self.action_queue = queue.Queue(maxsize=1)
        self.data_queue = queue.Queue(maxsize=1)
        # Thread for running the environment.
        self.thread = None
        # Dicts for keeping track of rewards etc, inside the env thread.
        self._thr_dones = {}
        self._thr_infos = {}
        self._thr_rewards = {}
        # store latest query id for debugging
        self._mn_query_id = 0
        # signal for thread to stop
        self._thread_stop_signal = threading.Event()
        # self._reset = False

    def run(self):
        """Override this to implement the multi-agent run loop.
        This should step through an episode, using the following methods:
        - self._thr_get_actions(obs) to query the policy
        - self._thr_log_rewards(rew)
        - self._thr_log_infos(infos)
        - self._thr_log_dones(dones)
        - self._end_episode(obs) to end the episode. This *must* be called to allow the final env.step() call to finish!"""
        raise NotImplementedError

    def _run(self):
        try:
            self.run()
        except SystemExit:
            pass

    def _thr_get_actions(self, observations, hidden: bool = False):
        """Query one or more agents for actions."""
        # Set a unique ID to ensure we get the response to this same query - avoid bugs.
        id = uuid.uuid4().hex

        # Set dones["__all__"]
        if "__all__" not in self._thr_dones:
            if all(self._thr_dones.values()) and len(self._thr_dones) > 0:
                self._thr_dones["__all__"] = True
            else:
                self._thr_dones["__all__"] = False

        if hidden:
            for agent in observations:
                if agent not in self._thr_infos:
                    self._thr_infos[agent] = {}
                self._thr_infos[agent]["hidden"] = True

        # Send observations, rewards, etc. to the RL training loop.
        self.data_queue.put(
            (
                id,
                observations,
                {agent: self._thr_rewards.pop(agent, 0) for agent in observations},
                {agent: self._thr_dones.pop(agent, False) for agent in list(observations.keys()) + ["__all__"]},
                {agent: self._thr_infos.pop(agent, {}) for agent in observations},
            )
        )
        # Reset rewards and infos
        # self._thr_rewards = {}
        # self._thr_infos = {}
        # Wait for actions
        while True:
            if not self.action_queue.empty():
                break
            if self._thread_stop_signal.is_set():
                # If the episode thread is meant to be stopped, we exit.
                # Raising SystemExit inside a thread will be silently ignored.
                raise SystemExit
            time.sleep(0.01)
        actions = self.action_queue.get()
        # Check that the ID matches
        assert actions[0] == id
        # Return actions
        return actions[1]

    def _thr_log_rewards(self, rew):
        """Log reward for an agent. Accumulates until next call to _thr_get_actions()."""
        for agent in rew:
            if agent in self._thr_rewards:
                self._thr_rewards[agent] += rew[agent]
            else:
                self._thr_rewards[agent] = rew[agent]

    def _thr_log_info(self, info):
        """Log info. Per-agent dict gets updated until next call to _thr_get_actions(). Dicts within dicts are overwritten."""
        for agent in info:
            if agent in self._thr_infos:
                self._thr_infos[agent].update(info[agent])
            else:
                self._thr_infos[agent] = info[agent]

    def _thr_set_dones(self, dones):
        """Set agent to be done."""
        for agent in dones:
            self._thr_dones[agent] = dones[agent]

    def _thr_end_episode(self, observations):
        """End the current episode."""
        for agent in self._thr_dones:
            self._thr_dones[agent] = True
        self._thr_dones["__all__"] = True
        # Set a unique ID to ensure we get the response to this same query - avoid bugs.
        id = uuid.uuid4().hex
        # Send observations, rewards, etc. to the RL training loop.
        self.data_queue.put((id, observations, self._thr_rewards, self._thr_dones, self._thr_infos))

    def reset(self):
        """Reset the environment."""
        # Try to stop the env thread
        if self.thread is not None:
            self._thread_stop_signal.set()
            self.thread.join(timeout=10)
            # Check that the episode has ended.
            assert self.thread.is_alive() == False, "Env thread did not stop within 10 seconds."
        # Start a new episode thread.
        self._thread_stop_signal.clear()
        self._thr_dones = {}
        self._thr_infos = {}
        self._thr_rewards = {}
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        # Wait for initial observation from episode thread
        (id, obs, rew, done, info) = self.data_queue.get()
        self._mn_query_id = id
        return obs
        # Slight hack here: we return empty on reset, so we can return an info dict even with the first observation on step().
        # self._reset = True
        # return {}

    def close(self):
        """Close the environment."""
        # Try to stop the env thread
        if self.thread is not None:
            self._thread_stop_signal.set()
            self.thread.join(timeout=10)
            # Check that the episode has ended.
            assert self.thread.is_alive() == False, "Env thread did not stop within 10 seconds."

    def step(self, actions):
        """Step the environment"""
        # Hacky workaround for hidden-queries experiments
        # if len(actions) > 0 and not self._reset:
        # Send actions to the episode thread
        # self.action_queue.put((self._mn_query_id, actions))
        # else:
        #     self._reset = False
        # Regular non-hacky version:
        self.action_queue.put((self._mn_query_id, actions))
        # Wait for observations from episode thread
        (id, obs, rew, done, info) = self.data_queue.get()
        # Update last query id
        self._mn_query_id = id
        # Return observations etc
        return obs, rew, done, info


class ThreadedMultiAgentWrapper(ThreadedMultiAgentEnv, MultiAgentWrapper):
    def __init__(self, env: MultiAgentEnv, identifier: Optional[str] = None):
        # TODO figure out init order etc - should MAWrapper call super init too?
        ThreadedMultiAgentEnv.__init__(self)
        MultiAgentWrapper.__init__(self, env, identifier)
        # super().__init__()

    def run(self):
        """Override this to implement the multi-agent run loop.
        This should step through an episode, using the methods below.
        Additionally, you must call self.env.step() at the beginning of each episode to reset the wrapped env.
        - self._thr_get_actions(obs) to query the policy
        - self._thr_log_rewards(rew)
        - self._thr_log_info(infos)
        - self._thr_log_dones(dones)
        - self._end_episode(obs) to end the episode. This *must* be called to allow the final env.step() call to finish!"""
        raise NotImplementedError

    def close(self):
        """Close the environment."""
        # Close my own env thread.
        super().close()
        # Close inner env
        self.env.close()
