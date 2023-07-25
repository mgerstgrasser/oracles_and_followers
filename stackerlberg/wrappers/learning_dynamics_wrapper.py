from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from stackerlberg.core.envs import MultiAgentWrapper

# Not used in the O&F paper.
# Not working with current RLLib (as of Jan 2023).
# Two issues:
# 1. RLLib does not like many episodes for one agent and only a single epside for the other.
#    This wrapper fixes this by renaming followers.
# 2. RLLib as of right now does not support different learning algorithms or even different update frequencies for different agents.


class LearningDynamicsAgentCopiesWrapper(MultiAgentWrapper):
    """This wrapper implements observed learning dynamics by "concatenating" many follower-episodes into one leader-episodes.
    If follower agents are controlled by a fast-converging learning algorithm,
    they should achieve equilibrium by the end of the leader episodse"""

    def __init__(self, env: MultiAgentEnv, leader_agent_id: str, n_follower_episodes: int = 1):
        super().__init__(env)
        self.leader = leader_agent_id
        self.n_follower_episodes = n_follower_episodes
        self.followers = self._agent_ids - {leader_agent_id}
        self._agent_ids = {leader_agent_id}
        self.action_space = {leader_agent_id: env.action_space[leader_agent_id]}
        self.observation_space = {leader_agent_id: env.observation_space[leader_agent_id]}
        for follower in self.followers:
            for n in range(n_follower_episodes):
                self._agent_ids.add(follower + "__COPY__" + str(n))
                self.action_space[follower + "__COPY__" + str(n)] = env.action_space[follower]
                self.observation_space[follower + "__COPY__" + str(n)] = env.observation_space[follower]
        self.action_space = spaces.Dict(self.action_space)
        self.observation_space = spaces.Dict(self.observation_space)

    def reset(self):
        # Reset follower episode counter
        self.cur_follower_episode = 0
        obs = self.env.reset()
        # Rename all the followers to the copy name.
        for follower in self.followers:
            if follower in obs:
                obs[follower + "__COPY__" + str(self.cur_follower_episode)] = obs[follower]
                del obs[follower]
        # obs should now be leader + follower_0__COPY__0, follower_1__COPY__0, ...
        return obs

    def _step_with_follower_renamed(self, actions, n):
        # Rename copies back to original names
        for follower in self.followers:
            if follower + "__COPY__" + str(n) in actions:
                actions[follower] = actions[follower + "__COPY__" + str(n)]
                del actions[follower + "__COPY__" + str(n)]
        # Call inner env step
        obs, rewards, dones, infos = self.env.step(actions)
        # Rename all the followers to the copy name.
        for follower in self.followers:
            if follower in obs:
                obs[follower + "__COPY__" + str(n)] = obs[follower]
                del obs[follower]
            if follower in rewards:
                rewards[follower + "__COPY__" + str(n)] = rewards[follower]
                del rewards[follower]
            if follower in dones:
                dones[follower + "__COPY__" + str(n)] = dones[follower]
                del dones[follower]
            if follower in infos:
                infos[follower + "__COPY__" + str(n)] = infos[follower]
                del infos[follower]
        return obs, rewards, dones, infos

    def step(self, actions):
        obs, rewards, dones, infos = self._step_with_follower_renamed(actions, self.cur_follower_episode)
        # Set leader reward to 0 for follower query steps:
        if self.cur_follower_episode < self.n_follower_episodes - 1:
            rewards[self.leader] = 0
        # Handle inner env episode end.
        # But only if we're not also at the end of the entire meta-episode!
        # We overwrite the leader observation with the one from self.env.reset().
        # Otherwise we would need to have two separate steps(), one to return the leader observation from the final step()
        # of the previous episode, and one to return the leader observation from the reset() of the next episode.
        # RLLib would have the leader return an action, too, which would be meaningless.
        # It's unclear if it is OK to drop the final observation.
        # For PG algorithms, it should be, for Q-learning, it might not be? But not clear. TODO check.
        if dones["__all__"] is True and self.cur_follower_episode < self.n_follower_episodes - 1:
            for follower in self.followers:
                dones[follower + "__COPY__" + str(self.cur_follower_episode)] = True
            dones[self.leader] = False
            dones["__all__"] = False
            self.cur_follower_episode += 1
            rst_obs = self.env.reset()
            # Rename all the followers to the copy name.
            for follower in self.followers:
                if follower in rst_obs:
                    obs[follower + "__COPY__" + str(self.cur_follower_episode)] = rst_obs[follower]
            obs[self.leader] = rst_obs[self.leader]

        return obs, rewards, dones, infos


class LearningDynamicsInfoWrapper(MultiAgentWrapper):
    """This wrapper implements observed learning dynamics by "concatenating" many follower-episodes into one leader-episodes.
    If follower agents are controlled by a fast-converging learning algorithm,
    they should achieve equilibrium by the end of the leader episodse"""

    def __init__(self, env: MultiAgentEnv, leader_agent_id: str, n_follower_episodes: int = 1, n_reward_episodes: int = 1):
        super().__init__(env)
        self.leader = leader_agent_id
        self.followers = self._agent_ids - {leader_agent_id}
        self.n_follower_episodes = n_follower_episodes
        self.n_reward_episodes = n_reward_episodes

    def reset(self):
        # Reset follower episode counter
        self.cur_follower_episode = 0
        obs = self.env.reset()
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        # Set leader reward to 0 for follower query steps:
        if self.cur_follower_episode < self.n_follower_episodes - self.n_reward_episodes:
            rewards[self.leader] = 0
        # Handle inner env episode end.
        # But only if we're not also at the end of the entire meta-episode!
        # We overwrite the leader observation with the one from self.env.reset().
        # Otherwise we would need to have two separate steps(), one to return the leader observation from the final step()
        # of the previous episode, and one to return the leader observation from the reset() of the next episode.
        # RLLib would have the leader return an action, too, which would be meaningless.
        # It's unclear if it is OK to drop the final observation.
        # For PG algorithms, it should be, for Q-learning, it might not be? But not clear. TODO check.
        if dones["__all__"] is True and self.cur_follower_episode < self.n_follower_episodes - 1:
            for follower in self.followers:
                if follower in infos and isinstance(infos[follower], dict):
                    infos[follower]["episode_done"] = True
                else:
                    infos[follower] = {"episode_done": True}
                dones[follower] = False
            dones[self.leader] = False
            dones["__all__"] = False
            self.cur_follower_episode += 1
            rst_obs = self.env.reset()
            obs[self.leader] = rst_obs[self.leader]

        return obs, rewards, dones, infos
