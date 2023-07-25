import threading

from stackerlberg.envs.matrix_game import MatrixGameEnv
from stackerlberg.wrappers.observed_queries_wrapper import ObservedQueriesWrapper


def test_threaded_env():
    """This creates a threaded env and repeatedly resets it, and closes it, to test that threads get closed properly."""
    env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True)
    env = ObservedQueriesWrapper(
        env,
        leader_agent_id="agent_0",
        queries={"q1": 1, "q2": 2, "q3": 3, "q4": 4},
        n_samples=1,
        samples_summarize="list",
    )
    env.close()
    for _ in range(5):
        env = MatrixGameEnv("prisoners_dilemma", episode_length=10, memory=True)
        env = ObservedQueriesWrapper(
            env,
            leader_agent_id="agent_0",
            queries={"q1": 1, "q2": 2, "q3": 3, "q4": 4},
            n_samples=1,
            samples_summarize="list",
        )
        for k in range(5):
            env.reset()
        env.close()
    assert threading.active_count() <= 5, "Too many threads, some must not have closed properly."
