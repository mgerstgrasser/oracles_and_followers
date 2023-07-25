import os
from collections import OrderedDict

from ray import tune
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.algorithms.a3c import A3C, a3c_torch_policy
from ray.rllib.algorithms.dqn import DQN, DQNTorchPolicy
from ray.rllib.algorithms.es import ES, ESTorchPolicy
from ray.rllib.algorithms.pg import PG, PGTorchPolicy
from ray.rllib.algorithms.ppo import PPO, PPOTF1Policy, PPOTorchPolicy
from ray.rllib.algorithms.sac import SAC, SACTorchPolicy
from ray.rllib.algorithms.simple_q import SimpleQ, SimpleQTorchPolicy
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from stackerlberg.envs.matrix_game import named_matrix_games
from stackerlberg.models.custom_fully_connected_torch_network import (
    CustomFullyConnectedNetwork,
)
from stackerlberg.models.linear_torch_model import LinearTorchModel
from stackerlberg.train.experiments.debug_callbacks import *
from stackerlberg.train.policies import (
    AlwaysCoop,
    AlwaysDefect,
    IPD_MostlyTFT,
    IPD_TFT_Coop_Defect,
    IPDCoopOrDefectPerEpisode,
    IPDRandomEveryEpisodePolicy,
    SmIPD_TFT_Coop_Defect,
)
from stackerlberg.trainers.stackerlberg_trainable_es import stackerlberg_trainable_es

experiment_configurations = {
    # --- All matrices, big figure in main text --- #
    "ipd_allmatrices_ppo_pg": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": tune.grid_search(list(named_matrix_games.keys())),
                    "discrete_obs": True,
                    "small_memory": False,
                    "episode_length": 10,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "seed": tune.grid_search([1, 2, 3, 4, 5]),
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PPO,
            "leader_policy_class": PPOTorchPolicy,
            "leader_config": {
                "lr": 0.008,
                "entropy_coeff": 0.0,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 1000,
                "train_batch_size": 1000,
                "sgd_minibatch_size": 1000,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "pre_training_stop_on_optimal": True,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 500,
            "post_training_iterations": 50,
            "randomize_leader": True,
            "pretrain_save_checkpoint": "auto",
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
    },
    "ipd_allmatrices_ppo_tabularq": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_tabularq",
                "env_config": {
                    "matrix_name": tune.grid_search(list(named_matrix_games.keys())),
                    "discrete_obs": True,
                    "small_memory": False,
                    "episode_length": 10,
                    "memory": True,
                    "n_q_episodes": 80,
                    "q_alpha": 0.01,
                    "reset_between_episodes": True,
                },
                "batch_mode": "complete_episodes",
            },
            # "deterministic_leader": True,
            # "deterministic_follower": True,
            "seed": tune.grid_search([1, 2, 3, 4, 5]),
            "leader_algorithm": PPO,
            "leader_policy_class": PPOTorchPolicy,
            "leader_config": {
                "lr": 0.03,
                "entropy_coeff": 0.0,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 8000,
                "train_batch_size": 8000,
                "sgd_minibatch_size": 8000,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_agent_ids": [],
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 0,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 2000,
            "post_training_iterations": 0,
            # "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "stop_condition": {"leader_results/timesteps_total": 2000000},
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "entropy_coeff": tune.uniform(0.0, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.0001,
                },
            },
            {
                "leader_config": {
                    "lr": 0.001,
                },
            },
            {
                "leader_config": {
                    "lr": 0.01,
                },
            },
            {
                "leader_config": {
                    "lr": 0.1,
                },
            },
            {
                "leader_config": {
                    "lr": 1.0,
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 4,
    },
    # --------------------- #
    # --- Basic BotS matrix game, used to show that not resetting follower between episodes can break things
    "bots_pg_tabularq": {
        "configuration": {
            "common_config": {
                "env": "matrix_game_tabularq",
                "env_config": {
                    "matrix": [[[2, 1], [0, 0]], [[0, 0], [1, 2]]],
                    "n_q_episodes": 10,
                    "hidden_queries": False,
                    "reset_between_episodes": True,
                    "param_noise": True,
                },
                "batch_mode": "complete_episodes",
            },
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PG,
            "leader_policy_class": PGTorchPolicy,
            "leader_config": {
                "lr": 0.156,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_agent_ids": [],
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 0,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 500,
            "post_training_iterations": 0,
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.015,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.004,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.03,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.012,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 4,
        "hyperopt_total_samples": 20,
    },
    # --- BotS, showing that out-of-eq leader reward leads to wrong eq
    "bots_dqn_tabularq_out_of_eq": {
        "configuration": {
            "common_config": {
                "env": "matrix_game_tabularq",
                "env_config": {
                    "matrix": [[[tune.grid_search([2, 1.5]), 0.001], [-5, 0]], [[-5, 0], [1, 2]]],
                    "n_q_episodes": 10,
                    "hidden_queries": False,
                    "reset_between_episodes": True,
                    "param_noise": True,
                    "reward_offset": 0.0,
                    "leader_reward_during_q": tune.grid_search([True, False]),
                    "q_init_zero": True,
                    "q_alpha": 0.2,
                },
                "batch_mode": "complete_episodes",
            },
            "seed": tune.grid_search(list(range(10))),
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "min_sample_timesteps_per_iteration": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "evaluation_config": {
                    "env_config": {
                        "_is_eval_env": True,
                    }
                },
                "lr": 0.1,
                "rollout_fragment_length": 10,
                "batch_mode": "complete_episodes",
                "train_batch_size": 1024,
                "learning_starts": 100,
                "exploration_config": {
                    "type": "ParameterNoise",
                    "random_timesteps": 0,
                    "initial_stddev": 1.0,
                    "sub_exploration": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 0.0,
                        "final_epsilon": 0.0,
                        "epsilon_timesteps": 1000000,
                    },
                    # "type": "EpsilonGreedy",
                    # "initial_epsilon": 1.0,
                    # "final_epsilon": 0.2,
                    # "epsilon_timesteps": 1000000,
                },
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_agent_ids": [],
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 0,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 200,
            "post_training_iterations": 0,
            # "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.0008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.0002,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.002,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.015,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 4,
    },
    # --- Show that leader memory can break things
    "smipd_leadermemory_pg_pg": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                    "tell_leader": True,
                },
                "batch_mode": "complete_episodes",
            },
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PG,
            "leader_policy_class": PGTorchPolicy,
            "leader_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 2000,
            "post_training_iterations": 50,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
                "config": [lambda **kwargs: kwargs["pretrain_config"]["env_config"].update({"tell_leader_mock": True})],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.015,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.03,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.004,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 10,
    },
    "smipd_leadernomemory_pg_pg": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                    "tell_leader": False,
                },
                "batch_mode": "complete_episodes",
            },
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PG,
            "leader_policy_class": PGTorchPolicy,
            "leader_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 2000,
            "post_training_iterations": 50,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
                "config": [lambda **kwargs: kwargs["pretrain_config"]["env_config"].update({"tell_leader_mock": True})],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.015,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.03,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.004,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 10,
    },
    # --- Show that queries hidden from leader can break things
    "smipd_hiddenqueries_pg_pg_new": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix": [
                        [[4, 3], [2, 4]],
                        [[3, 1], [1, 2]],
                    ],
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                    "hidden_queries": tune.grid_search([True, False]),
                },
                "batch_mode": "complete_episodes",
            },
            "seed": tune.grid_search(list(range(10))),
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PG,
            "leader_policy_class": PGTorchPolicy,
            "leader_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 2000,
            "post_training_iterations": 50,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
                "post-leader": [
                    lambda **kwargs: kwargs["results"].update(
                        {
                            f"leader_action_{i}": kwargs["leader_trainer"].compute_single_action(i, policy_id="agent_0", explore=False)
                            for i in range(3)
                        }
                    ),
                ],
            },
            "log_weights": True,
            # "pretrain_load_checkpoint": os.path.dirname(os.path.realpath(__file__)) + "/pretrain_checkpoint_prisoners_dilemma_0.pkl",
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 0.001),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.0002,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.0008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.00005,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 10,
    },
    # --- Show that hidden queries do work for non-RL approaches such as ES
    "ipd_ipd_ppo_pg_savecheckpoint": {
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "seed": tune.grid_search([1, 2, 3, 4, 5]),
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": PPO,
            "leader_policy_class": PPOTorchPolicy,
            "leader_config": {
                "lr": 0.008,
                "entropy_coeff": 0.0,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 1000,
                "train_batch_size": 1000,
                "sgd_minibatch_size": 1000,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "pre_training_stop_on_optimal": True,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 500,
            "post_training_iterations": 50,
            "randomize_leader": True,
            "pretrain_save_checkpoint": "auto",
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
    },
    "smipd_es_pg_new": {
        # Works now, but ugly
        "configuration": {
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    # "matrix_name": "prisoners_dilemma",
                    "matrix": [
                        [[4, 3], [2, 4]],
                        [[3, 1], [1, 2]],
                    ],
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                    "hidden_queries": tune.grid_search([True, False]),
                },
                "batch_mode": "complete_episodes",
            },
            "seed": tune.grid_search([1, 2, 3]),
            "deterministic_leader": True,
            "deterministic_follower": True,
            "leader_algorithm": ES,
            "leader_policy_class": ESTorchPolicy,
            "leader_config": {
                "num_workers": 1,
                "lr": tune.grid_search([0.015, 0.004]),
                "rollout_fragment_length": 100,
                "train_batch_size": 1000,
                "learning_starts": 0,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.02,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 0,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 2000,
            "post_training_iterations": 50,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "leader_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "leader_config": {
                    "lr": 0.002,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.015,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "leader_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "leader_results/evaluation/policy_reward_mean/agent_0",
        "hyperopt_seeds": 10,
        "trainable": tune.with_resources(stackerlberg_trainable_es, tune.PlacementGroupFactory([{"CPU": 1.0}] + [{"CPU": 1.0}] * 1)),
    },
    # --- Tests ---
    # ---------- testing save and load checkpoints --
    "test_save_load_checkpoint": {
        "configuration": {
            # Tests follower against 1/3 each always-coop, always-defect, and TFT leader.
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "lr": 0.1,
                "train_batch_size": 10,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                # "train_batch_size": 256,
                # "sgd_minibatch_size": 256,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 1,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 0,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
            "pretrain_save_checkpoint": "./pretrain_checkpoint.pkl",
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
    },
    # ---------- basic tests --
    "matrix_bots": {
        "configuration": {
            "pre_training_iterations": 4,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 16,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "common_config": {
                "env": "matrix_game_stackelberg_observed_queries",
                "env_config": {"matrix": [[[1.0, 0.5], [0, 0]], [[0, 0], [0.5, 1.0]]], "reward_offset": 0},
                "framework": "torch",
                "rollout_fragment_length": 1,
                "train_batch_size": 256,
                "min_sample_timesteps_per_iteration": 16,
                "lr": 0.008,
                "replay_buffer_config": {
                    "learning_starts": 0,
                },
            },
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 4,
                "evaluation_duration_unit": "episodes",
                "exploration_config": {
                    # "type": "ParameterNoise",
                    # "random_timesteps": 0,
                    # "initial_stddev": 1.0,
                    # "sub_exploration": {
                    #     "type": "EpsilonGreedy",
                    #     "initial_epsilon": 0.0,
                    #     "final_epsilon": 0.0,
                    #     "epsilon_timesteps": 1000000,
                    # },
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.1,
                    "epsilon_timesteps": 256,
                },
                "lr_schedule": [[0, 0.008], [256, 0.00001]],
            },
            "leader_algorithm": SimpleQ,
            "follower_algorithm": SimpleQ,
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                },
            },
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                },
            },
            "randomize_leader": True,
            "callbacks": {
                "post-pretrain": [
                    lambda **kwargs: kwargs["results"].update(
                        {
                            "action_0": kwargs["pre_trainer"].compute_single_action(
                                OrderedDict(original_space=0, none_0=0), policy_id="agent_1", explore=False
                            ),
                            "action_1": kwargs["pre_trainer"].compute_single_action(
                                OrderedDict(original_space=0, none_0=1), policy_id="agent_1", explore=False
                            ),
                        }
                    ),
                ],
            },
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] == 1.0
        and results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_1"] == 0.5,
    },
    "matrix_bots_pretrainonly": {
        "configuration": {
            "pre_training_iterations": 1,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 12,
            "outer_iterations": 0,
            "post_training_iterations": 0,
            "common_config": {
                "env": "matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix": [[[1, 0.5], [0, 0]], [[0, 0], [0.5, 1]]],
                },
                "framework": "torch",
                "rollout_fragment_length": 1,
                "train_batch_size": 256,
                "min_sample_timesteps_per_iteration": 64,
                "lr": 0.008,
                "replay_buffer_config": {
                    "learning_starts": 0,
                },
            },
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 4,
                "evaluation_duration_unit": "episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                },
            },
            "follower_algorithm": SimpleQ,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                },
            },
            "randomize_leader": True,
            "callbacks": {
                "post-pretrain": [
                    lambda **kwargs: kwargs["results"].update(
                        {
                            "action_0": kwargs["pre_trainer"].compute_single_action(
                                OrderedDict({"original_space": 0, "none_0": 0}), policy_id="agent_1", explore=False
                            ),
                            "action_1": kwargs["pre_trainer"].compute_single_action(
                                OrderedDict({"original_space": 0, "none_0": 1}), policy_id="agent_1", explore=False
                            ),
                        }
                    ),
                ],
            },
        },
        "success_condition": lambda results: results["action_0"] == 0 and results["action_1"] == 1,
    },
    "smipd_dqn_pg_pretrain": {
        "configuration": {
            # Tests follower against 1/3 each always-coop, always-defect, and TFT leader.
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "lr": 0.1,
                "train_batch_size": 10,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                # "train_batch_size": 256,
                # "sgd_minibatch_size": 256,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 500,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "follower_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "follower_config": {
                    "lr": 0.008,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.004,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.002,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.001,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.0005,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "follower_best_responds/sum",
        "hyperopt_seeds": 10,
    },
    "smipd_dqn_ppo_pretrain": {
        "configuration": {
            # Tests follower against 1/3 each always-coop, always-defect, and TFT leader.
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 10,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "lr": 0.1,
                "train_batch_size": 10,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PPO,
            "follower_policy_class": PPOTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "lr": 0.803,
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 8000,
                "train_batch_size": 8000,
                "sgd_minibatch_size": 2000,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 50,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "follower_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "follower_config": {
                    "lr": 0.02,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.1,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.5,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "follower_best_responds/sum",
        "hyperopt_seeds": 10,
    },
    "smipd_dqn_pg_nondiscrete_pretrain": {
        "configuration": {
            # Tests follower against 1/3 each always-coop, always-defect, and TFT leader.
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": False,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "lr": 0.1,
                "train_batch_size": 10,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": PG,
            "follower_policy_class": PGTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [
                        24,
                    ],
                    "vf_share_layers": True,
                    "custom_model_config": {
                        "bias": False,
                    },
                    "custom_model": CustomFullyConnectedNetwork,
                },
            },
            "follower_config": {
                "lr": 0.008,
                "min_sample_timesteps_per_iteration": 100,
                # "train_batch_size": 256,
                # "sgd_minibatch_size": 256,
                "metrics_smoothing_episodes": 1,
                "rollout_fragment_length": 100,
                "train_batch_size": 100,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "learning_starts": 0,
            },
            "pre_training_iterations": 75,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "follower_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"final_epsilon": tune.loguniform(0.01, 1.0)},
            },
        },
        "hyperopt_startingpoints": [
            {
                "follower_config": {
                    "lr": 0.02,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.1,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
            {
                "follower_config": {
                    "lr": 0.5,
                    # "exploration_config": {"final_epsilon": 0.2},
                },
            },
        ],
        "hyperopt_metric": "follower_best_responds/sum",
        "hyperopt_seeds": 10,
    },
    "smipd_dqn_dqn_pretrain": {
        "configuration": {
            # Tests follower against 1/3 each always-coop, always-defect, and TFT leader.
            "common_config": {
                "env": "repeated_matrix_game_stackelberg_observed_queries",
                "env_config": {
                    "matrix_name": "prisoners_dilemma",
                    "discrete_obs": True,
                    "small_memory": True,
                    "episode_length": 5,
                    "memory": True,
                },
                "batch_mode": "complete_episodes",
            },
            "leader_algorithm": SimpleQ,
            "leader_policy_class": SimpleQTorchPolicy,
            "leader_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "lr": 0.1,
                "train_batch_size": 10,
            },
            "leader_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_algorithm": SimpleQ,
            "follower_policy_class": SimpleQTorchPolicy,
            "follower_policy_config": {
                "model": {
                    "fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "custom_model": LinearTorchModel,
                },
            },
            "follower_config": {
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "min_sample_timesteps_per_iteration": 100,
                "metrics_smoothing_episodes": 1,
                "lr": 0.008,
                "train_batch_size": 1024,
                "rollout_fragment_length": 100,
                "batch_mode": "complete_episodes",
                "learning_starts": 0,
                "exploration_config": {
                    "type": "ParameterNoise",
                    "random_timesteps": 0,
                    "initial_stddev": 1.0,
                    "sub_exploration": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 0.0,
                        "final_epsilon": 0.0,
                        "epsilon_timesteps": 1000000,
                    },
                    # "type": "EpsilonGreedy",
                    # "initial_epsilon": 1.0,
                    # "final_epsilon": 0.2,
                    # "epsilon_timesteps": 1000000,
                },
            },
            "pre_training_iterations": 5000,
            "inner_iterations_follower": 0,
            "inner_iterations_leader": 1,
            "outer_iterations": 1,
            "post_training_iterations": 0,
            "randomize_leader": True,
            # "_debug_dont_train_leader": True,
            "callbacks": {
                "post-pretrain": [smipd_check_follower_best_response],
            },
            "log_weights": True,
        },
        "success_condition": lambda results: results["leader_results"]["evaluation"]["policy_reward_mean"]["agent_0"] >= 6.0,
        "hyperopt_searchspace": {
            "follower_config": {
                "lr": tune.loguniform(0.00001, 1.0),
                # "rollout_fragment_length": tune.choice([1, 2, 4, 8, 16, 32, 64]),
                # "exploration_config": {"sub_exploration": {"initial_epsilon": tune.loguniform(0.001, 1.0)}},
            },
        },
        "hyperopt_startingpoints": [
            {
                "follower_config": {
                    "lr": 0.002,
                    # "exploration_config": {"sub_exploration": {"initial_epsilon": 0.1}},
                },
            },
            {
                "follower_config": {
                    "lr": 0.02,
                    # "exploration_config": {"sub_exploration": {"initial_epsilon": 0.1}},
                },
            },
            {
                "follower_config": {
                    "lr": 0.2,
                    # "exploration_config": {"sub_exploration": {"initial_epsilon": 0.1}},
                },
            },
        ],
        "hyperopt_metric": "follower_best_responds/sum",
        "hyperopt_seeds": 4,
    },
}
