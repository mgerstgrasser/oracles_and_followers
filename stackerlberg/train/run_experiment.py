import argparse
import os
from typing import *

import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.air.config import RunConfig
from ray.tune.search import Repeater
from ray.tune.search.hyperopt import HyperOptSearch

from stackerlberg.train.experiments.configurations import experiment_configurations
from stackerlberg.trainers.stackerlberg_trainable import stackerlberg_trainable
from stackerlberg.utils.utils import update_recursively


def run_experiment(
    experiment: str = "matrix_bots",
    ray_num_cpus: int = 0,
    ray_local_mode: bool = False,
    wandb_project: str = "test_local",
    wandb_group: str = "test_local",
    hyperopt: bool = False,
    seed: int = 0,
    no_tune: bool = False,
):
    """This does hyperparam optimisation for a two-stage curriculum learning workflow on a trivial matrix game."""

    # Set up Ray
    # We set local mode for debugging.
    # we ignore reinit error in case we run from pytest, which reuses processes, so ray may already be initialized.
    ray.init(local_mode=ray_local_mode, ignore_reinit_error=True, num_cpus=ray_num_cpus or None, include_dashboard=False)

    # Get the config dict from experiments/configurations.py, and the experiment name
    config = experiment_configurations[experiment]["configuration"]
    trainable = experiment_configurations[experiment].get("trainable", stackerlberg_trainable)

    if not no_tune:
        if wandb_project == "auto":
            wandb_project = experiment
        if wandb_group == "auto":
            wandb_group = experiment

        # Set up Weights And Biases logging if API key is set in environment variable.
        if "WANDB_API_KEY" in os.environ:
            callbacks = [
                WandbLoggerCallback(
                    project=wandb_project,
                    # project="test_local",
                    api_key=os.environ["WANDB_API_KEY"],
                    log_config=True,
                    resume=False,
                    # name="test_ipd_cluster",
                    # dir="./ray_results/wandb/",
                    group=wandb_group,
                )
            ]
        else:
            callbacks = []
            print("WARNING! No wandb API key found, running without wandb!")

        if hyperopt and all(
            [key in experiment_configurations[experiment] for key in ["hyperopt_searchspace", "hyperopt_metric", "hyperopt_startingpoints"]]
        ):
            # Run hyperopt search from config in experiments/configurations.py
            # We take the default config, and update it with the hyperopt search space.
            # Initial parameter configurations have to be set in the config dict too, same for metric etc.
            config = update_recursively(config, experiment_configurations[experiment]["hyperopt_searchspace"])
            current_best_configs = experiment_configurations[experiment]["hyperopt_startingpoints"]
            mode = experiment_configurations[experiment].get("hyperopt_mode", "max")
            metric = experiment_configurations[experiment]["hyperopt_metric"]
            seeds = experiment_configurations[experiment].get("hyperopt_seeds", 4)
            total_samples = experiment_configurations[experiment].get("hyperopt_total_samples", 1000)
            stop_condition = experiment_configurations[experiment].get("hyperopt_stop_condition", None)
            hyperopt_search = HyperOptSearch(metric=metric, mode=mode, points_to_evaluate=current_best_configs)
            re_search_alg = Repeater(hyperopt_search, repeat=seeds, set_index=True)
            tune_config = tune.TuneConfig(mode=mode, metric=metric, num_samples=total_samples, search_alg=re_search_alg, reuse_actors=False)
        else:
            # Run a single experiment from config in experiments/configurations.py
            if hyperopt:
                print("WARNING! Experiment does not have hyperopt configuration, running without hyperopt!")
            tune_config = tune.TuneConfig(reuse_actors=False)
            # We set the seed here, so that we can reproduce the experiment.
            if not "seed" in config:
                config["__trial_index__"] = seed
                config["seed"] = seed
            stop_condition = experiment_configurations[experiment].get("stop_condition", None)

        tuner = tune.Tuner(
            trainable,
            param_space=config,
            run_config=RunConfig(
                name=f"{wandb_project}__{wandb_group}",
                callbacks=callbacks,
                local_dir="./ray_results",
                stop=stop_condition,
            ),
            tune_config=tune_config,
        )

        results = tuner.fit()
    else:
        for results in trainable(config):
            print(results)

    print("Done!")
    if "success_condition" in experiment_configurations[experiment]:
        if experiment_configurations[experiment]["success_condition"](results._experiment_analysis.trials[0].last_result) == True:
            print("Success!")
        else:
            print("Warning: Results did not pass success condition in config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for Stackelberg RL", add_help=False)
    parser.add_argument(
        "--ray_num_cpus",
        type=int,
        default=0,
        help="Number of CPU cores to run on",
    )
    parser.add_argument(
        "--ray_local_mode",
        action="store_true",
        help="If enabled, init ray in local mode.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="If enabled, just call trainable, without tune.",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="If enabled, do hyperopt search.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="matrix_bots",
        help="Name of the experiment config to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to run (only relevant if hyperopt is disabled)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test_local",
        help="Name of the wandb project",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="test_local",
        help="Name of the wandb group",
    )
    cli_args, remaining_cli = parser.parse_known_args()
    run_experiment(
        experiment=cli_args.experiment,
        ray_num_cpus=cli_args.ray_num_cpus,
        ray_local_mode=cli_args.ray_local_mode,
        wandb_project=cli_args.wandb_project,
        wandb_group=cli_args.wandb_group,
        hyperopt=cli_args.hyperopt,
        seed=cli_args.seed,
        no_tune=cli_args.no_tune,
    )
