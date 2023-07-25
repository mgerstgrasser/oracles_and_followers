from gym import spaces

from stackerlberg.core.envs import ThreadedMultiAgentEnv, ThreadedMultiAgentWrapper


class ThreadedTestEnv(ThreadedMultiAgentEnv):
    def __init__(self, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self._agent_ids = {f"agent_{n}" for n in range(0, self.num_agents)}
        self.action_space = spaces.Dict({agent: spaces.Discrete(2) for agent in self._agent_ids})
        self.observation_space = spaces.Dict({agent: spaces.Discrete(2) for agent in self._agent_ids})

    def run(self):
        print(f"Starting threaded episode")
        for _ in range(10):
            print(f"Stepping threaded episode")
            actions = self._thr_get_actions({f"agent_0": 0, f"agent_1": 0})
            if actions[f"agent_0"] == 0:
                self._thr_log_rewards({f"agent_0": 1})
            if actions[f"agent_1"] == 1:
                self._thr_log_rewards({f"agent_1": 1})
            if self._thread_stop_signal.is_set():
                print(f"Stopping threaded episode early on signal")
                break
        self._thr_end_episode({f"agent_0": 0, f"agent_1": 0})
        print(f"Ended threaded episode")
        return None


class ThreadedTestWrapper(ThreadedMultiAgentWrapper):
    def run(self):
        print(f"Starting threaded wrapper episode")
        obs = self.env.reset()
        while True:
            print(f"Stepping threaded wrapper episode")
            actions = self._thr_get_actions(obs)
            obs, rew, done, info = self.env.step(actions)
            self._thr_log_rewards(rew)
            self._thr_log_info(info)
            self._thr_set_dones(done)
            if done["__all__"]:
                break
        print(f"Stopping threaded wrapper episode")
        self._thr_end_episode(obs)
