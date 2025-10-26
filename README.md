# CLUTCH (previously called RACER) 

## RL Race Car Assistant

This is a reinforcement learning project to train DQN and PPO agents to be
high-level decision-making assistants for a race car driver, in addition to getting
long-term strategy via gemini 2.5-flash-lite and elevenlabs for audio output. This features
independent dashboards and backends along with a comparison front-end to compare model performance
on a track.

### Project Structure

DQN

* `train.py`: The main script to run for training the agent.
* `agent.py`: Contains the `DQNAgent` and Q-Network (PyTorch model).
* `environment.py`: Contains the `RaceCarEnvironment` (the physics simulator).
* `config.py`: Holds all hyperparameters, track constants, and reward values.
* `track_data.json`: The raw data for the track (radius, pit stops).

PPO

* `train.py`: The main script to run for training the agent.
* `f1_env.py`: Contains the `F1Car` agent/sim representation and `F1Env` Gymnasium RL environment.
* `solver.py`: Contains the RK-4 ODE solver for continuous simulation.
* `track.py`: Contains the `Track` class for track representation and generation while training
* `train.py`: Training script with visual evals
* `track_**.json`: The raw data for the track (x, y, radius, pit stops).

### Setup

1.  Install dependencies:
    `pip install -r requirements.txt`

2.  Define Gemini and ElevenLabs API keys in lines 918/919 of comparison.html
    
### How to Train

1.  Navigate to `ppo/` and/or `dqn/` and run:
    `python train.py`
2.  The trained model will be saved as `.pth`file for DQN and `.zip` file for PPO and a graph
    of the training scores of DQN will be saved as `training_scores.png`.

### How to Run

1.  Setup backend: cd into `ppo/` and then run `python InferenceBackend.py` for PPO. Similarly, for DQN, cd into `dqn/` and run `python DQNBackend.py`
2.  Frontend: Run `python -m http.server 5500` 

#### Dashboards: 
Go to  [`localhost:5500/ppo`](http://localhost:5500/ppo) for PPO and [`localhost:5500/dqn`](http://localhost:5500/dqn) for DQN on your web browser

#### Track Racing Comparison (Run RL agents against each other and obtain high level insights):
Go to [`localhost:5500/comparison.html`](http://localhost:5500/comparison.html) on your web browser
