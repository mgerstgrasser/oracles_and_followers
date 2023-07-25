from gym import spaces

from stackerlberg.train.make_env import make_repeated_matrix_observed_queries_env


def test_repeated_matrix_observed_queries_env():
    env = make_repeated_matrix_observed_queries_env(
        episode_length=10,
        discrete_obs=True,
        small_memory=True,
    )
    assert env.observation_space["agent_0"] == spaces.Discrete(3)
    assert env.observation_space["agent_1"] == spaces.Discrete(24)
    env.close()
