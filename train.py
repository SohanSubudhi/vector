# train.py
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Make sure mph_to_mps is available for the logger
from environment import RaceCarEnvironment, mps_to_mph # Import mps_to_mph
from agent import DQNAgent
import config

def train():
    env = RaceCarEnvironment()
    agent = DQNAgent()

    scores = []
    scores_window = deque(maxlen=100)  # for avg score
    epsilon = config.EPSILON_START

    num_episodes = 21 # You can set this higher for full training

    print("Starting training...")

    for i_episode in range(1, num_episodes + 1):
        print(f"\n--- Starting Episode {i_episode} ---") # Keep episode start print

        state = env.reset()
        episode_score = 0
        done = False
        step_count = 0

        # Flag to print details for this episode
        log_details = (i_episode == 1 or i_episode % 10 == 0) # Log every episode
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0} # Accel, Brake, Coast, Pit

        while not done:
            # Select action
            action = agent.act(state, epsilon)
            action_counts[action] += 1

            # --- CORRECTED Pit Stop Log ---
            if action == 3 and log_details:
                 # Check the *actual* pit conditions from the environment
                 is_in_pit_zone = env.distance_on_lap >= env.pit_entry_start_distance and \
                                  env.distance_on_lap < env.pit_entry_end_distance

                 is_at_safe_speed = mps_to_mph(env.speed_mps) < config.MAX_PIT_ENTRY_SPEED_MPH

                 if is_in_pit_zone and is_at_safe_speed:
                    print(f"  [Ep {i_episode}, Step {step_count}] *** PIT STOP ATTEMPT (SUCCESS) ***")
            # --- END CORRECTION ---

            # Take action in environment
            next_state, reward, done = env.step(action)

            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state
            episode_score += reward
            step_count += 1

            # Perform one step of the optimization
            agent.learn()

            # --- CORRECTED Log details every 200 steps ---
            if log_details and (step_count % 200 == 0):
                current_lap = config.TOTAL_LAPS - env.laps_remaining + 1
                # Use max() to prevent negative lap numbers if something goes wrong
                current_lap = max(1, current_lap)
                print(f"  [Ep {i_episode}, Step {step_count}] "
                      f"Lap: {current_lap}/{config.TOTAL_LAPS} | "
                      f"Prog: {env.distance_on_lap:.0f}m / {env.lap_distance:.0f}m | "
                      f"Speed: {mps_to_mph(env.speed_mps):.1f} mph | " # Use mps_to_mph
                      f"V_max: {mps_to_mph(env.current_v_max_mps):.1f} mph | " # Use mps_to_mph
                      f"Fuel: {env.fuel:.1f}L | "
                      f"Avg Wear: {np.mean(env.tire_wear):.3f}")
            # --- END CORRECTION ---

            if done:
                break

        scores_window.append(episode_score)
        scores.append(episode_score)

        # Update epsilon
        epsilon = max(config.EPSILON_END, epsilon - (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY)

        # Update target network
        if i_episode % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()

        # --- CORRECTED End of Episode Report ---
        episode_end_reason = "Unknown (check max steps?)" # More informative default
        if env.laps_remaining <= 0:
            episode_end_reason = "RACE FINISHED"
        elif env.fuel <= 0.0:
            episode_end_reason = "OUT OF FUEL"
        # Optional: Add a check for max steps if you implement one

        # Print progress every 10 episodes
        if i_episode % 10 == 0:
            print(f"--- Episode {i_episode} Finished ---")
            print(f"  End Reason: {episode_end_reason}")
            print(f"  Total Steps: {step_count}")
            print(f"  Total Score: {episode_score:.2f}")
            # Ensure window isn't empty before calculating mean
            avg_score = np.mean(scores_window) if scores_window else 0.0
            print(f"  Avg Score (Last {len(scores_window)}): {avg_score:.2f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"    Action Counts: {action_counts}")
        # --- END CORRECTION ---

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
    # plt.show() # Often better to save than show in Colab

if __name__ == "__main__":
    train()