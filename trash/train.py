import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import math
from pyURControl import ur_control
from operator import add
from math import pi 
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyURControl import ur_control
import math
import time
import roboticstoolbox as rtb
a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]

# Link twists (alpha)
alpha = [pi/2, 0.0, 0.0, pi/2, -pi/2, 0.0]

# Link offsets (d)
d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]

# Create the list of links
links = []
for i in range(6):
    link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i])
    links.append(link)

# Initialize the UR3e robot model
ur3e = rtb.DHRobot(links, name='UR3e')
# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

# Define the PPO Agent
class PPOAgent:
    def __init__(self, input_dim, output_dim, gamma=0.99, epsilon=0.2, lr=3e-4):
        self.actor_critic = ActorCritic(input_dim, output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self.actor_critic(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, next_value, masks):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            gae = delta + self.gamma * self.epsilon * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        log_probs_old = torch.tensor(log_probs_old)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        action_probs, values = self.actor_critic(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = (returns - values).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Initialize the environment and agent
def initialize_environment():
    ur_control.init('192.168.56.101')
    ur_control.power_on()
    ur_control.break_release()

def get_current_state():
    joint_angles = ur_control.get_position()
    return joint_angles

def get_random_target_position(radius=500):
    """
    Generate a random target position within the UR3e's spherical workspace.

    Args:
        radius (float): The radius of the workspace in mm.

    Returns:
        np.ndarray: A 3D vector representing the target position in mm.
    """
    # Generate random spherical coordinates
    r = radius * (np.random.rand() ** (1/3))  # Cube root to ensure uniform distribution
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = np.random.uniform(0, np.pi)       # Polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.array([x, y, z])

def check_done(state, target_position, distance_threshold=0.05, max_distance=5.0):
    """
    Determine if the episode should terminate based on the distance to the target position.
    
    Args:
        state (np.ndarray): The current joint angles of the robot.
        target_position (np.ndarray): The target Cartesian position.
        distance_threshold (float): The distance within which the target is considered reached.
        max_distance (float): The maximum allowable distance before considering the robot too far.
        
    Returns:
        bool: True if the episode should terminate, False otherwise.
        int: A termination code indicating the reason.
    """
    end_effector_position = ur3e.fkine(state).t
    distance = np.linalg.norm(end_effector_position - target_position)
    
    if distance < distance_threshold:
        return True, 1  # Reached target
    elif distance > max_distance:
        return True, -1  # Moved too far away
    return False, 0  # Continue episode

def calculate_reward(state, target_position, distance_threshold=0.05, max_distance=5.0):
    """
    Calculate the reward based on the distance to the target position.
    
    Args:
        state (np.ndarray): The current joint angles of the robot.
        target_position (np.ndarray): The target Cartesian position.
        distance_threshold (float): The distance within which the target is considered reached.
        max_distance (float): The maximum allowable distance before considering the robot too far.
        
    Returns:
        float: The calculated reward.
    """
    end_effector_position = ur3e.fkine(state).t
    distance = np.linalg.norm(end_effector_position - target_position)
    
    if distance < distance_threshold:
        return 2.0  # High reward for reaching the target
    elif distance > max_distance:
        return -2.0  # Negative reward for moving too far away
    else:
        return -distance  # Negative reward proportional to distance


# Training loop
def train():
    input_dim = 6  # Number of joints
    output_dim = 8  # Number of actions (joint velocities)
    agent = PPOAgent(input_dim, output_dim)

    initialize_environment()

    target_position = get_random_target_position()

    for episode in range(1000):
        state = get_current_state()
        done = False
        episode_rewards = []
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        next_state = None
        next_value = 0
        count = 0
        while not done:
            action, log_prob = agent.get_action(state)
            next_state = state + action  # Update state based on action
            reward = calculate_reward(next_state, target_position)
            done, termination_code = check_done(next_state, target_position)
            
            # Handle termination codes if needed
            if termination_code == 1:
                print("Target reached!")
            elif termination_code == -1:
                print("Moved too far away!")

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(next_value)
            rewards.append(reward)

            state = next_state
            next_value = reward  # Update next_value for advantage calculation
            count += 1
            if done or count == 1000:
                break

        advantages = agent.compute_advantages(rewards, values, next_value, [1] * len(rewards))
        returns = [adv + val for adv, val in zip(advantages, values)]

        agent.update(states, actions, log_probs, returns, advantages)

        print(f"Episode {episode + 1}: Reward = {sum(rewards)}")

if __name__ == "__main__":
    train()
