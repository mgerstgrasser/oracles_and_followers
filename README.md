# Oracles and Followers: Stackelberg Equilibria in Deep Multi-Agent Reinforcement Learning

This repository contains code for the ICML 2023 paper "Oracles & Followers: Stackelberg Equilibria in Deep Multi-Agent Reinforcement Learning". arXiv version available at <https://arxiv.org/abs/2210.11942>

## Installation

Set up and activate a Python virtual environment using `pyenv virtualenv` or Anaconda, using python version 3.10.5.
Then, run `pip install -e .` to install our code, as well as all necessary dependencies.

## Running Experiments

To run experiments, use the `stackerlberg/train/run_experiment.py` script. To run each of the experiments in the paper, use the following commands:

```bash
# All 12 matrices
python stackerlberg/train/run_experiment.py --experiment=ipd_allmatrices_pg_pg
python stackerlberg/train/run_experiment.py --experiment=ipd_allmatrices_ppo_pg
# Hidden vs observed queries
python stackerlberg/train/run_experiment.py --experiment=smipd_hiddenqueries_pg_pg
python stackerlberg/train/run_experiment.py --experiment=smipd_nothiddenqueries_pg_pg
python stackerlberg/train/run_experiment.py --experiment=smipd_hiddenqueries_dqn_pg
python stackerlberg/train/run_experiment.py --experiment=smipd_nothiddenqueries_dqn_pg
# Invariant leader vs non-invariant
python stackerlberg/train/run_experiment.py --experiment=smipd_leadermemory_pg_pg
python stackerlberg/train/run_experiment.py --experiment=smipd_leadernomemory_pg_pg
```

You can additionally specify `--ray_num_cpus=X` to limit the number of CPU cores the program will use, and you can specify `--hyperopt` to try a variety of learning rates (this was used for the hidden/observed queries and invariant leader experiments). If you set an API key for weights and biases as the environment variable `WANDB_API_KEY`, the results will be logged to weights and biases.

Specifically for the hidden-queries experiment using DQN, a small change to rllib internal code is necessary, detailed in the next section.

## RLlib changes

Inside your Python library folder, in the file ray/rllib/evaluation/postprocessing.py
change line 132 and following to

```python
    if not isinstance(rollout[Postprocessing.ADVANTAGES], np.ndarray):
        rollout[Postprocessing.ADVANTAGES] = np.array(
            rollout[Postprocessing.ADVANTAGES].tolist()
        )
```

This is only needed for the experiment using DQN for the leader.
