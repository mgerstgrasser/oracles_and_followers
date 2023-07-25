# import gym.spaces

# from stackerlberg.envs.matrix_game import MatrixGameEnv
# from stackerlberg.wrappers.repeated_matrix_hypernetwork import (
#     RepeatedMatrixHypernetworkWrapper,
# )


# def test_hypernetwork():
#     """Test the matrix game without memory, and checks that the correct reward is returned."""
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.Box)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [0, 1, 1, 0, 0],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"] == 2
#     assert rew["agent_0"] == 0
#     assert rew["agent_1"] == -3
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 1,
#         }
#     )
#     assert obs["agent_1"] == 3
#     assert rew["agent_0"] == -3
#     assert rew["agent_1"] == 0
#     env.close()
#     # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
#     # 1, 2: agent 1 action 0, 3, 4 action 1
#     # 1, 3: agent 0 action 0, 2, 4 action 1


# def test_hypernetwork_discrete():
#     """Test the matrix game without memory, and checks that the correct reward is returned."""
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env, discrete=True)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.MultiBinary)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [1, 0, 0, 1, 1],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"] == 2
#     assert rew["agent_0"] == 0
#     assert rew["agent_1"] == -3
#     while not don["__all__"]:
#         obs, rew, don, inf = env.step(
#             {
#                 f"agent_1": 0,
#             }
#         )
#         assert obs["agent_1"] == 1
#         assert rew["agent_0"] == -1
#         assert rew["agent_1"] == -1
#     assert obs["agent_0"] == 0
#     env.close()
#     # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
#     # 1, 2: agent 1 action 0, 3, 4 action 1
#     # 1, 3: agent 0 action 0, 2, 4 action 1


# def test_hypernetwork_queries():
#     """Test the matrix game without memory, and checks that the correct reward is returned."""
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env, queries=True)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.Box)
#     assert isinstance(env.observation_space["agent_1"], gym.spaces.Dict)
#     assert isinstance(env.observation_space["agent_1"]["queries"], gym.spaces.Box)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [0, 1, 1, 0, 0],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 2, "queries": [0, 1, 1, 0, 0]}
#     assert rew["agent_0"] == 0
#     assert rew["agent_1"] == -3
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 1,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 3, "queries": [0, 1, 1, 0, 0]}
#     assert rew["agent_0"] == -3
#     assert rew["agent_1"] == 0
#     env.close()
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env, queries=True)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.Box)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [0.0, 0.1, 0.3, 0.5, 1.0],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"]["queries"] == [0.0, 0.1, 0.3, 0.5, 1.0]
#     env.close()
#     # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
#     # 1, 2: agent 1 action 0, 3, 4 action 1
#     # 1, 3: agent 0 action 0, 2, 4 action 1


# def test_hypernetwork_queries_discrete():
#     """Test the matrix game without memory, and checks that the correct reward is returned."""
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env, queries=True, discrete=True)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.MultiBinary)
#     assert isinstance(env.observation_space["agent_1"], gym.spaces.Dict)
#     assert isinstance(env.observation_space["agent_1"]["queries"], gym.spaces.MultiBinary)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [1, 0, 0, 1, 1],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 2, "queries": [1, 0, 0, 1, 1]}
#     assert rew["agent_0"] == 0
#     assert rew["agent_1"] == -3
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 1,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 3, "queries": [1, 0, 0, 1, 1]}
#     assert rew["agent_0"] == -3
#     assert rew["agent_1"] == 0
#     env.close()


# def test_hypernetwork_smallmem_queries_discrete():
#     """Test the matrix game without memory, and checks that the correct reward is returned."""
#     env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True, small_memory=True, reward_offset=-4)
#     env = RepeatedMatrixHypernetworkWrapper(env, queries=True, discrete=True)
#     assert isinstance(env.action_space["agent_0"], gym.spaces.MultiBinary)
#     assert isinstance(env.observation_space["agent_1"], gym.spaces.Dict)
#     assert isinstance(env.observation_space["agent_1"]["queries"], gym.spaces.MultiBinary)
#     obs = env.reset()
#     assert obs["agent_0"] == 0
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_0": [1, 0, 0],
#         }
#     )
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 0,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 2, "queries": [1, 0, 0]}
#     assert rew["agent_0"] == 0
#     assert rew["agent_1"] == -3
#     obs, rew, don, inf = env.step(
#         {
#             f"agent_1": 1,
#         }
#     )
#     assert obs["agent_1"] == {"original_space": 1, "queries": [1, 0, 0]}
#     assert rew["agent_0"] == -3
#     assert rew["agent_1"] == 0
#     env.close()
