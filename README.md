# vector
hacktx_2025

## RL Race Car Assistant

This is a reinforcement learning project to train a DQN agent to be a
high-level decision-making assistant for a race car driver.

### Project Structure

* `train.py`: The main script to run for training the agent.
* `agent.py`: Contains the `DQNAgent` and Q-Network (PyTorch model).
* `environment.py`: Contains the `RaceCarEnvironment` (the physics simulator).
* `config.py`: Holds all hyperparameters, track constants, and reward values.
* `track_data.json`: The raw data for the track (radius, pit stops).

### How to Run

1.  Install dependencies:
    `pip install torch numpy matplotlib`

2.  Start the training:
    `python train.py`

3.  The trained model will be saved as `race_car_dqn.pth` and a graph
    of the training scores will be saved as `training_scores.png`.