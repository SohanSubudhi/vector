# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import config

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition tuple for replay buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        # Ensure transitions are stored in the same order as the namedtuple
        # Expected order: (state, action, reward, next_state, done)
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    """Deep Q-Network Model"""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self):
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayBuffer(config.MEMORY_SIZE)
        self.steps_done = 0

    def act(self, state, epsilon):
        """Choose an action using epsilon-greedy policy."""
        if random.random() > epsilon:
            # Exploit
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_t)
                # Select action with max Q-value
                return q_values.max(1)[1].view(1, 1).item()
        else:
            # Explore
            # Action 3 (pit) is only valid near pit entry,
            # but for simplicity, we let the agent explore it.
            # The environment will penalize it if it's a bad choice.
            return random.randrange(self.action_dim)

    def learn(self):
        """Update the Q-network using a batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert lists of arrays to single numpy arrays first (avoids slow tensor creation)
        state_np = np.array(batch.state)
        next_state_np = np.array(batch.next_state)
        action_np = np.array(batch.action, dtype=np.int64)
        reward_np = np.array(batch.reward, dtype=np.float32)
        done_np = np.array(batch.done, dtype=np.bool_)

        state_batch = torch.from_numpy(state_np).float().to(device)
        action_batch = torch.from_numpy(action_np).long().unsqueeze(1).to(device)
        reward_batch = torch.from_numpy(reward_np).float().to(device)
        next_state_batch = torch.from_numpy(next_state_np).float().to(device)
        done_batch = torch.from_numpy(done_np).to(device)

        # 1. Compute Q(s_t, a)
        # These are the Q-values our policy_net *predicted*
        # for the actions we *actually took*.
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. Compute V(s_t+1) = max_a Q_target(s_t+1, a)
        # These are the *max* Q-values for the next state,
        # according to the stable target_net.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # 3. Compute the expected Q-value (target)
        # If state was terminal (done=True), its value is just the reward.
        # Otherwise, it's reward + gamma * next_q_value
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # 4. Compute loss
        # Use Smooth L1 Loss (Huber Loss)
        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        # 5. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """Copy weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())