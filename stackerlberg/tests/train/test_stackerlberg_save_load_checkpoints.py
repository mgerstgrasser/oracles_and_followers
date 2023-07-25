import copy

import pytest
import ray

import stackerlberg.train.make_env
from stackerlberg.train.experiments.configurations import experiment_configurations
from stackerlberg.trainers.stackerlberg_trainable import stackerlberg_trainable


def test_save_load_checkpoint():
    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

    # Create a checkpoint
    config = copy.deepcopy(experiment_configurations["test_save_load_checkpoint"]["configuration"])

    def get_weights(pre_trainer=None, results={}, **kwargs):
        results["follower_weights"] = pre_trainer.get_weights()["agent_1"]

    config["callbacks"]["post-pretrain"] = [
        get_weights,
    ]
    config["seed"] = 123
    results = list(stackerlberg_trainable(config))

    # Load the checkpoint
    def get_weights_2(leader_trainer=None, results={}, **kwargs):
        follower_weights = leader_trainer.get_weights()["agent_1"]
        results["follower_weights"] = follower_weights

    config2 = copy.deepcopy(experiment_configurations["test_save_load_checkpoint"]["configuration"])
    config2["pretrain_load_checkpoint"] = "./pretrain_checkpoint.pkl"
    config2["callbacks"]["post-leader"] = [
        get_weights_2,
    ]
    config2["pre_training_iterations"] = 0
    config2["inner_iterations_leader"] = 1
    config2["seed"] = 456
    results_2 = list(stackerlberg_trainable(config2))

    # Check that the weights are the same
    weights_before = results[-1]["follower_weights"]
    weights_after = results_2[-1]["follower_weights"]
    assert (weights_before["_model.weight"] == weights_after["_model.weight"]).all(), f"Test failed, {weights_before} != {weights_after}"
    assert (weights_before["_value.weight"] == weights_after["_value.weight"]).all(), f"Test failed, {weights_before} != {weights_after}"
