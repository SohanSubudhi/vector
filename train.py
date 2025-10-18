# train.py
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from environment import RaceCarEnvironment
from agent import DQNAgent
import config

def train():
    env = RaceCarEnvironment()
    agent = DQNAgent()
    
    scores = []
    scores_window = deque(maxlen=100)  # for avg score
    epsilon = config.EPSILON_START
    
    num_episodes = 2000
    
    print("Starting training...")
    
    for i_episode in range(1, num_episodes + 1):
        print(f"episode {i_episode}")
        state = env.reset()
        episode_score = 0
        done = False
        
        while not done:
            print("about to act")
            # Select action
            action = agent.act(state, epsilon)
            
            print("about to step")
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            print("replay buffer next")
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            episode_score += reward
            
            print("about to learn")
            # Perform one step of the optimization
            agent.learn()
            
            if done:
                break
        
        print("appending scores")
        scores_window.append(episode_score)
        scores.append(episode_score)
        
        print("updating epsilon")
        # Update epsilon
        epsilon = max(config.EPSILON_END, epsilon - (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY)
        
        print("updating target network")
        # Update target network
        if i_episode % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()
            
        # Print progress
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.3f}')
            
    print("\nTraining finished.")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'race_car_dqn.pth')
    print("Model saved to race_car_dqn.pth")
    
    # Plot scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Score over Time')
    plt.savefig('training_scores.png')
    plt.show()

if __name__ == "__main__":
    train()