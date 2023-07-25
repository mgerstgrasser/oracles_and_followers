import numpy as np
from ray.tune.registry import register_env as ray_register_env

from stackerlberg.envs.matrix_game import MatrixGameEnv, StochasticRewardWrapper
from stackerlberg.envs.test_envs import ThreadedTestEnv, ThreadedTestWrapper
from stackerlberg.wrappers.action_to_dist_wrapper import ActionToDistWrapper
from stackerlberg.wrappers.dict_to_discrete_obs_wrapper import DictToDiscreteObsWrapper
from stackerlberg.wrappers.learning_dynamics_wrapper import LearningDynamicsInfoWrapper
from stackerlberg.wrappers.observed_queries_wrapper import ObservedQueriesWrapper
from stackerlberg.wrappers.repeated_matrix_hypernetwork import (
    RepeatedMatrixHypernetworkWrapper,
)
from stackerlberg.wrappers.tabularq_wrapper import TabularQWrapper

registered_environments = {}


def register_env(name_or_function=None):
    """Decorator for registering an environment.
    Registeres the decorated function as a factory function for environments.
    Does this for both our own registry as well as rllib's."""
    if callable(name_or_function):
        # If we got a callable that means the decorator was called without paranthesis, i.e. @register
        # In that case we directly wrap the function
        n = name_or_function.__name__
        registered_environments[n] = name_or_function

        def env_creator_kwargs(env_config):
            return name_or_function(**env_config)

        ray_register_env(n, env_creator_kwargs)
        return name_or_function
    else:
        # Else we should have gotten a name string, so we return a decorator.
        def _register_env(function):
            if name_or_function is None:
                n = function.__name__
            else:
                n = name_or_function
            registered_environments[n] = function

            def env_creator_kwargs(env_config):
                return function(**env_config)

            ray_register_env(n, env_creator_kwargs)
            return function

        return _register_env


@register_env("test_env")
def make_test_env(
    num_agents: int = 2,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    env = ThreadedTestEnv(num_agents)
    env = ThreadedTestWrapper(env)
    return env


@register_env("matrix_game")
def make_matrix_env(
    episode_length: int = 1,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length)
    return env


@register_env("matrix_game_stackelberg_learning_dynamics")
def make_matrix_sld_env(
    n_follower_episodes: int = 32,
    n_reward_episodes: int = 4,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix)
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    if not _is_eval_env:
        env = LearningDynamicsInfoWrapper(
            env, leader_agent_id="agent_0", n_follower_episodes=n_follower_episodes, n_reward_episodes=n_reward_episodes
        )
    return env


@register_env("matrix_game_stackelberg_observed_queries")
def make_matrix_observed_queries_env(
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    reward_offset: float = 0.0,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, reward_offset=reward_offset)
    env = ObservedQueriesWrapper(
        env, leader_agent_id="agent_0", queries={"none": 0}, n_samples=n_samples, samples_summarize=samples_summarize
    )
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    return env


@register_env("matrix_game_tabularq")
def make_matrix_tabularq_env(
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    n_q_episodes: int = 50,
    q_alpha: float = 0.1,
    reset_between_episodes: bool = True,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    follower_sparse_reward_prob: int = 1,
    follower_sparse_reward_scale: int = 1,
    follower_sparse_reward_deterministic: bool = False,
    leader_reward_during_q: bool = False,
    param_noise: bool = False,
    q_init_zero: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, reward_offset=0)
    if follower_sparse_reward_prob != 1:
        env = StochasticRewardWrapper(
            env,
            prob=follower_sparse_reward_prob,
            scale=follower_sparse_reward_scale,
            deterministic=follower_sparse_reward_deterministic,
        )
    env = TabularQWrapper(
        env,
        leader_agent_id="agent_0",
        follower_agent_id="agent_1",
        n_q_episodes=n_q_episodes,
        reset_between_episodes=reset_between_episodes,
        epsilon=0.1,
        alpha=q_alpha,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
        param_noise=param_noise,
        leader_reward_during_q=leader_reward_during_q if not _is_eval_env else False,
        q_init_zero=q_init_zero,
    )
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs and tell_leader:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    return env


@register_env("repeated_matrix_game")
def make_repeated_matrix_env(
    episode_length: int = 10,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
    return env


@register_env("repeated_matrix_game_tabularq")
def make_repeated_matrix_tabularq_env(
    episode_length: int = 10,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    n_q_episodes: int = 50,
    reset_between_episodes: bool = True,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.9,
    q_init_zero: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    env = TabularQWrapper(
        env,
        leader_agent_id="agent_0",
        follower_agent_id="agent_1",
        n_q_episodes=n_q_episodes,
        reset_between_episodes=reset_between_episodes,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
        q_init_zero=q_init_zero,
    )
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs and tell_leader:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    return env


@register_env("repeated_matrix_game_stackelberg_observed_queries")
def make_repeated_matrix_observed_queries_env(
    episode_length: int = 10,
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == [] or matrix == () or len(matrix) == 0:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    if small_memory:
        qu = {"q0": 0, "q1": 1, "q2": 2}
    else:
        qu = {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    env = ObservedQueriesWrapper(
        env,
        leader_agent_id="agent_0",
        queries=qu,
        n_samples=n_samples,
        samples_summarize=samples_summarize,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
    )
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
        if tell_leader:
            env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env)
    return env


@register_env("repeated_matrix_game_stackelberg_learning_dynamics")
def make_repeated_matrix_sld_env(
    episode_length: int = 10,
    n_follower_episodes: int = 1,
    n_reward_episodes: int = 1,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    discrete_obs: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == [] or matrix == () or len(matrix) == 0:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True)
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    if not _is_eval_env:
        env = LearningDynamicsInfoWrapper(
            env, leader_agent_id="agent_0", n_follower_episodes=n_follower_episodes, n_reward_episodes=n_reward_episodes
        )
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
    return env
