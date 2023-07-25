from stackerlberg.envs.matrix_game import MatrixGameEnv


def test_matrix_game():
    """Test the matrix game without memory, and checks that the correct reward is returned."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], reward_offset=0)
    obs = env.reset()
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 0})
    assert rew["agent_0"] == 1
    assert rew["agent_1"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 0})
    assert rew["agent_0"] == 0
    assert rew["agent_1"] == 0
    env.close()


def test_matrix_game_memory():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, reward_offset=0)
    obs = env.reset()
    assert obs["agent_0"] == 0
    assert obs["agent_1"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 0})
    assert obs["agent_0"] == 1
    assert obs["agent_1"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 0})
    assert obs["agent_0"] == 2
    assert obs["agent_1"] == 2
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 1})
    assert obs["agent_0"] == 3
    assert obs["agent_1"] == 3
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 1})
    assert obs["agent_0"] == 4
    assert obs["agent_1"] == 4
    env.close()


def test_matrix_game_smallmemory():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, small_memory=True, reward_offset=0)
    obs = env.reset()
    assert obs["agent_0"] == 0
    assert obs["agent_1"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 0})
    assert obs["agent_0"] == 1
    assert obs["agent_1"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 0})
    assert obs["agent_0"] == 1
    assert obs["agent_1"] == 2
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 1})
    assert obs["agent_0"] == 2
    assert obs["agent_1"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 1})
    assert obs["agent_0"] == 2
    assert obs["agent_1"] == 2
    env.close()


def test_matrix_game_named():
    """Test prisoner's dilemma, and check that the payoffs are correct."""
    env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True)
    obs = env.reset()
    # Both cooperate
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 0})
    assert rew["agent_0"] == 0.5
    assert rew["agent_1"] == 0.5
    # Both defect
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 1})
    assert rew["agent_0"] == -0.5
    assert rew["agent_1"] == -0.5
    # Agent 0 cooperates, agent 1 defects, agent 0 gets punished
    obs, rew, don, inf = env.step({f"agent_0": 0, f"agent_1": 1})
    assert rew["agent_0"] == -1.5
    assert rew["agent_1"] == 1.5
    # Agent 1 cooperates, agent 0 defects, agent 1 gets punished
    obs, rew, don, inf = env.step({f"agent_0": 1, f"agent_1": 0})
    assert rew["agent_0"] == 1.5
    assert rew["agent_1"] == -1.5
    env.close()
