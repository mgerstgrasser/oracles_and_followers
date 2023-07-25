from cgitb import small

from stackerlberg.envs.matrix_game import MatrixGameEnv
from stackerlberg.wrappers.dict_to_discrete_obs_wrapper import DictToDiscreteObsWrapper
from stackerlberg.wrappers.observed_queries_wrapper import ObservedQueriesWrapper


def test_matrix_game_memory_obsq_discrete():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True)
    env = ObservedQueriesWrapper(
        env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}, samples_summarize="list"
    )
    env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
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
    # assert obs["agent_1"]["q0_0"] == 0
    # assert obs["agent_1"]["q1_0"] == 1
    # assert obs["agent_1"]["q2_0"] == 0
    # assert obs["agent_1"]["q3_0"] == 1
    # assert obs["agent_1"]["q4_0"] == 1
    # assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_1"] == 1 * 1 + 1 * 2 + 1 * 8
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 1, "agent_1": 0})
    assert obs["agent_0"] == 2
    assert obs["agent_1"] == 2 * 32 + 1 * 8 + 1 * 2 + 1 * 1
    env.close()


def test_matrix_game_smallmemory_obsq_discrete():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, small_memory=True)
    env = ObservedQueriesWrapper(env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2}, samples_summarize="list")
    env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
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
    # assert obs["agent_1"]["q0_0"] == 0
    # assert obs["agent_1"]["q1_0"] == 1
    # assert obs["agent_1"]["q2_0"] == 0
    # assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_1"] == 0 * 8 + 0 * 4 + 1 * 2 + 0 * 1
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 1, "agent_1": 0})
    # assert obs["agent_1"]["q0_0"] == 0
    # assert obs["agent_1"]["q1_0"] == 1
    # assert obs["agent_1"]["q2_0"] == 0
    # assert obs["agent_1"]["original_space"] == 2
    assert obs["agent_1"] == 2 * 8 + 0 * 4 + 1 * 2 + 0 * 1
    assert obs["agent_0"] == 1
    env.close()


def test_matrix_game_smallmemory_obsq_discrete_telleader():
    """Test the matrix game with memory, and checks that the obs is correct."""
    env = MatrixGameEnv([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], episode_length=10, memory=True, small_memory=True)
    env = ObservedQueriesWrapper(
        env, leader_agent_id="agent_0", queries={"q0": 0, "q1": 1, "q2": 2}, samples_summarize="list", tell_leader=True
    )
    env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
    env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    obs = env.reset()
    assert "agent_0" in obs
    assert "agent_1" not in obs
    assert obs["agent_0"] == 3
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert obs["agent_0"] == 4
    obs, rew, don, inf = env.step({f"agent_0": 1})
    assert obs["agent_0"] == 5
    obs, rew, don, inf = env.step({f"agent_0": 0})
    assert "agent_1" in obs
    # assert obs["agent_1"]["q0_0"] == 0
    # assert obs["agent_1"]["q1_0"] == 1
    # assert obs["agent_1"]["q2_0"] == 0
    # assert obs["agent_1"]["original_space"] == 0
    assert obs["agent_1"] == 0 * 8 + 0 * 4 + 1 * 2 + 0 * 1
    assert obs["agent_0"] == 0
    obs, rew, don, inf = env.step({f"agent_0": 1, "agent_1": 0})
    # assert obs["agent_1"]["q0_0"] == 0
    # assert obs["agent_1"]["q1_0"] == 1
    # assert obs["agent_1"]["q2_0"] == 0
    # assert obs["agent_1"]["original_space"] == 2
    assert obs["agent_1"] == 2 * 8 + 0 * 4 + 1 * 2 + 0 * 1
    assert obs["agent_0"] == 1
    env.close()
