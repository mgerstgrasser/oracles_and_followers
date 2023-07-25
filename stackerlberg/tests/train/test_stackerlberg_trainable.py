import pytest
import ray

import stackerlberg.train.make_env
from stackerlberg.train.experiments.configurations import experiment_configurations
from stackerlberg.trainers.stackerlberg_trainable import stackerlberg_trainable


def test_stackerlberg_trainable():
    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = {
        "inner_iterations_follower": 1,
        "inner_iterations_leader": 1,
        "outer_iterations": 1,
        "post_training_iterations": 1,
        "common_config": {
            "env": "matrix_game",
        },
    }
    results = list(stackerlberg_trainable(config))


def test_stackerlberg_trainable_randomized():
    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = {
        "inner_iterations_follower": 1,
        "inner_iterations_leader": 1,
        "outer_iterations": 1,
        "post_training_iterations": 1,
        "common_config": {
            "env": "matrix_game",
            "seed": 1,
        },
        "randomize_follower": True,
        "randomize_leader": True,
    }
    results = list(stackerlberg_trainable(config))


# Currently works most of the time, but fails for a small number of seeds
# @pytest.mark.slow
@pytest.mark.parametrize("seed", range(16))  # 128
def test_matrix_bots(seed):
    """This tests a two-stage curriculum learning workflow on a trivial matrix game."""

    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = experiment_configurations["matrix_bots"]["configuration"]
    config["__trial_index__"] = seed
    iter = stackerlberg_trainable(config)
    results = next(iter)
    for results in iter:
        continue

    assert experiment_configurations["matrix_bots"]["success_condition"](results), f"Test failed, {results}"


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(6))  # 128
def test_matrix_bots_pg_dqn(seed):
    """This tests a two-stage curriculum learning workflow on a trivial matrix game."""

    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = experiment_configurations["matrix_bots_pg_dqn"]["configuration"]
    config["__trial_index__"] = seed
    iter = stackerlberg_trainable(config)
    results = next(iter)
    for results in iter:
        continue

    assert experiment_configurations["matrix_bots"]["success_condition"](results), f"Test failed, {results}"


# Currently works most of the time, but fails for a small number of seeds
# @pytest.mark.slow
@pytest.mark.parametrize("seed", range(16))  # 128
def test_matrix_bots_pretrain(seed):
    """This tests a two-stage curriculum learning workflow on a trivial matrix game."""

    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = experiment_configurations["matrix_bots_pretrainonly"]["configuration"]
    config["__trial_index__"] = seed
    iter = stackerlberg_trainable(config)
    results = next(iter)
    for results in iter:
        continue

    assert experiment_configurations["matrix_bots_pretrainonly"]["success_condition"](results), f"Test failed, {results}"


# Currently works most of the time, but fails for a small number of seeds
@pytest.mark.slow
@pytest.mark.parametrize("seed", range(16))  # 16
def test_matrix_bots_ppoleader_pretrain(seed):
    """This tests a two-stage curriculum learning workflow on a trivial matrix game."""

    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    config = experiment_configurations["matrix_bots_ppoleader_pretrain"]["configuration"]
    config["__trial_index__"] = seed
    iter = stackerlberg_trainable(config)
    results = next(iter)
    for results in iter:
        continue

    assert experiment_configurations["matrix_bots_ppoleader_pretrain"]["success_condition"](results), f"Test failed, {results}"
