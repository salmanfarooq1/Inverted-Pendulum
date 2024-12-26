# Inverted Pendulum with Reinforcement Learning

## Description

This project demonstrates a reinforcement learning agent that learns to balance an inverted pendulum using the Proximal Policy Optimization (PPO) algorithm.

## Dependencies

The project uses the following key libraries:

*   gymnasium
*   stable-baselines3
*   pygame
*   numpy

The full list of dependencies, including specific versions, is in the `environment.yml` file.

## Installation (Using Conda)

1.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate rl-pendulum
    ```

## Training

To train the agent, run:

```bash
python train.py