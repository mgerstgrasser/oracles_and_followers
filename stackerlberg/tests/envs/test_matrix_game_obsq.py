from cgitb import small

from stackerlberg.envs.matrix_game import MatrixGameEnv
from stackerlberg.wrappers.observed_queries_wrapper import ObservedQueriesWrapper


def test_matrix_game_memory_obsq():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True)
    env = ObservedQueriesWrapper(env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4})
    obs = env.reset()
    assert "agent_0" in obs
    assert "agent_1" not in obs
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert obs["agent_0"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert obs["agent_0"] == 2
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert obs["agent_0"] == 3
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert obs["agent_0"] == 4
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert "agent_1" in obs
    assert obs["agent_1"]["q0_0"] == 0
    assert obs["agent_1"]["q1_0"] == 1
    assert obs["agent_1"]["q2_0"] == 0
    assert obs["agent_1"]["q3_0"] == 1
    assert obs["agent_1"]["q4_0"] == 1
    assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_0"] == 0
    env.close()


def test_matrix_game_smallmemory_obsq():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, small_memory=True)
    env = ObservedQueriesWrapper(env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2})
    obs = env.reset()
    assert "agent_0" in obs
    assert "agent_1" not in obs
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert obs["agent_0"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert obs["agent_0"] == 2
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert "agent_1" in obs
    assert obs["agent_1"]["q0_0"] == 0
    assert obs["agent_1"]["q1_0"] == 1
    assert obs["agent_1"]["q2_0"] == 0
    assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 1, "agent_1": 0})
    assert obs["agent_1"]["q0_0"] == 0
    assert obs["agent_1"]["q1_0"] == 1
    assert obs["agent_1"]["q2_0"] == 0
    assert obs["agent_1"]["original_space"] == 2
    assert obs["agent_0"] == 1
    env.close()


def test_matrix_game_smallmemory_obsq_tellleader():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, small_memory=True)
    env = ObservedQueriesWrapper(env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2}, tell_leader=True)
    obs = env.reset()
    assert "agent_0" in obs
    assert "agent_1" not in obs
    assert obs["agent_0"]["original_space"] == 0
    assert obs["agent_0"]["is_query"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert obs["agent_0"]["original_space"] == 1
    assert obs["agent_0"]["is_query"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert obs["agent_0"]["original_space"] == 2
    assert obs["agent_0"]["is_query"] == 1
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert "agent_1" in obs
    assert obs["agent_1"]["q0_0"] == 0
    assert obs["agent_1"]["q1_0"] == 1
    assert obs["agent_1"]["q2_0"] == 0
    assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_0"]["original_space"] == 0
    assert obs["agent_0"]["is_query"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 1, "agent_1": 0})
    assert obs["agent_1"]["q0_0"] == 0
    assert obs["agent_1"]["q1_0"] == 1
    assert obs["agent_1"]["q2_0"] == 0
    assert obs["agent_1"]["original_space"] == 2
    assert obs["agent_0"]["original_space"] == 1
    assert obs["agent_0"]["is_query"] == 0
    env.close()
