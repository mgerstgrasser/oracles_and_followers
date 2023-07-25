from stackerlberg.envs.test_envs import ThreadedTestEnv, ThreadedTestWrapper


def test_threaded_env():
    env = ThreadedTestEnv(2)
    obs = env.reset()
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.reset()
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    env.close()


def test_threaded_wrapper():
    env = ThreadedTestEnv(2)
    env = ThreadedTestWrapper(env)
    obs = env.reset()
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.reset()
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    obs = env.step({f"agent_0": 0, f"agent_1": 0})
    env.close()
