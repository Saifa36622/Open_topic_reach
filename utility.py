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
from spatialgeometry import Cylinder
from math import pi
import matplotlib.pyplot as plt
import os
from roboticstoolbox.backends.PyPlot import PyPlot

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Initialize previous joint positions and velocities
previous_q = None
previous_qd = None
initialized = False
DISTANCE_THRESHOLD = 0.1
UNUSED_TARGET = np.array([0.0, 0.0, 0.0])

def feedback(robot, dt):
    global previous_q, previous_qd, initialized

    current_q = robot.q

    if previous_q is None:
        previous_q = current_q
        previous_qd = np.zeros_like(current_q)
        initialized = True
        return current_q, np.zeros_like(current_q), np.zeros_like(current_q), robot.fkine(current_q).t, False

    qd = (current_q - previous_q) / dt
    qdd = (qd - previous_qd) / dt
    ee_pose = robot.fkine(current_q).t

    previous_q = current_q
    previous_qd = qd

    return current_q, qd, qdd, ee_pose, True


def reset(ur3e,dt):
    global previous_q, previous_qd, initialized

    previous_q = None
    previous_qd = None
    initialized = False

    ur3e.q = 6* [0.0]
    cmd = generate_cmd()
    timestep = [0]
    currrent_q, currrent_qqd, currrent_qqdd, end_pos,valid = feedback(ur3e, dt)
    state = np.concatenate((timestep,currrent_q,currrent_qqd ,currrent_qqdd,end_pos, cmd.flatten()))
    return state,cmd


def target_check_term(end_pos,cmd,current_timestep):
    
    for i in range(len(cmd)):
        target = cmd[i]
        distance = np.linalg.norm(end_pos - target)

        if distance < DISTANCE_THRESHOLD:
            
            cmd[i] = UNUSED_TARGET
            # print("cmd left",5-len(cmd))
            # Compute the reward based on how quickly the target was reached
            reward = (1000 - current_timestep)

            return reward , cmd
    return 0 ,cmd

def check_distance(end_pos,cmd):
    distance_sum = 0
    check = 0
    for target in cmd:
        if np.all(target == UNUSED_TARGET) :
            continue
        distance = np.linalg.norm(end_pos - target)
        distance_sum += distance
        check += 1
        # Assign a reward inversely proportional to the distance
        # You can adjust the scaling factor as needed

        # reward += 1.0 / (distance + 1e-6)
        return distance_sum

def check_done(current_timestep,cmd):
    check = 0
    for i in range(len(cmd)):
        target = cmd[i]
        if np.all(target == UNUSED_TARGET):
            check +=1 
            # print("cmd left", 5 - check)
    if check == 5 :
        return True , 10,check
    if current_timestep == 1000  : 
        return True , -10,check
    else :
        return False ,0 ,check
    
def velo_check(currrent_qqd,currrent_qqdd):

    weight_qd= 1e-10
    weight_qdd= 1e-11

    velocity_penalty = weight_qd * np.sum(np.square(currrent_qqd))
    acceleration_penalty = weight_qdd * np.sum(np.square(currrent_qqdd))

    # Total penalty is the sum of both components
    penalty = velocity_penalty + acceleration_penalty

    return penalty

def compute_reward(robot,cmd,current_timestep,dt): # and terminate 
    reward = 0
    done = False
    
    currrent_q, currrent_qqd, currrent_qqdd, end_pos,valid = feedback(robot, dt)

    done , finish_check ,num_sc = check_done(current_timestep,cmd)

    reward += finish_check

    if done :
        return reward ,done,cmd ,0,num_sc
    
    # check target term
    new_reward , new_cmd = target_check_term(end_pos,cmd,current_timestep)
    reward += new_reward

    # Near target term
    # reward += near_target(end_pos,cmd)
    distance = check_distance(end_pos,cmd)
    # velo check 
    reward -= velo_check(currrent_qqd,currrent_qqdd)

    return reward ,done,new_cmd ,distance,num_sc

def compute_next_q(q, qd, qd_prev, dt, max_acc):
    # Calculate the desired change in velocity
    delta_qd = qd - qd_prev

    # Compute the maximum allowable change in velocity
    max_delta_qd = max_acc * dt

    # Clip the change in velocity to the allowable range
    delta_qd_clipped = np.clip(delta_qd, -max_delta_qd, max_delta_qd)

    # Compute the new velocity
    qd_new = qd_prev + delta_qd_clipped

    # Update the joint positions
    q_new = q + qd_new * dt

    return q_new, qd_new

def generate_cmd():
    # Define limits for x, y, z (adjust as needed for your workspace)
    x_range = (-0.5, 0.5)
    y_range = (-0.5, 0.5)
    z_range = (0.2, 0.5)

    # Generate 5 target points with random x, y, z values
    targets = np.random.uniform(
        low=[x_range[0], y_range[0], z_range[0]],
        high=[x_range[1], y_range[1], z_range[1]],
        size=(5, 3)
    )

    return targets
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(128, 128)):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)
    

# # Define the PPO Agent
# class PPOAgent:
#     def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2):
#         self.actor = Actor(state_dim, action_dim)
#         self.critic = Critic(state_dim)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
#         self.gamma = gamma
#         self.eps_clip = eps_clip

#     def select_action(self, state):
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         action_mean = self.actor(state_tensor)
#         action_std = torch.ones_like(action_mean) * 0.1  # Define action_std appropriately
#         dist = torch.distributions.Normal(action_mean, action_std)
#         action = dist.sample()

#         action = torch.clamp(action, min=-1.0, max=1.0)  

#         log_prob = dist.log_prob(action).sum(dim=-1)
#         return action.detach().numpy(), log_prob.detach()

#     def compute_returns(self, rewards, dones, next_value):
#         returns = []
#         R = next_value
#         for reward, done in zip(reversed(rewards), reversed(dones)):
#             R = reward + self.gamma * R * (1.0 - float(done))  # Convert done to float to ensure proper math
#             returns.insert(0, R)
#         return torch.tensor(returns, dtype=torch.float32)

#     def update(self, trajectories):
#         states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32)
#         actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.float32)
#         old_log_probs = torch.tensor([t['log_prob'] for t in trajectories], dtype=torch.float32)
#         returns = torch.tensor(np.array([t['returns'] for t in trajectories]), dtype=torch.float32)



#         # Compute advantages
#         values = self.critic(states).squeeze()
#         advantages = returns - values.detach()

#         # Update actor
#         action_means = self.actor(states)
#         action_stds = torch.ones_like(action_means) * 0.1  # Adjust std as needed
#         dist = torch.distributions.Normal(action_means, action_stds)
#         log_probs = dist.log_prob(actions).sum(dim=-1)
#         ratios = torch.exp(log_probs - old_log_probs)
#         surr1 = ratios * advantages
#         surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#         actor_loss = -torch.min(surr1, surr2).mean()

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         values = self.critic(states).view(-1)
#         returns = returns.view(-1)
        
#         # Update critic
#         critic_loss = nn.MSELoss()(values, returns)
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         return actor_loss,critic_loss


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2, lam=0.95):
        self.actor = Actor(state_dim, action_dim).to('cuda')
        self.critic = Critic(state_dim).to('cuda')
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lam = lam  # lambda for GAE

    # def select_action(self, state):
    #     state_tensor = torch.tensor(state, dtype=torch.float32)
    #     action_mean = self.actor(state_tensor)
    #     action_std = torch.ones_like(action_mean) * 0.1  # Adjust as needed
    #     dist = torch.distributions.Normal(action_mean, action_std)
    #     raw_action = dist.sample()
    #     action = torch.tanh(raw_action)  # Squash action to (-1, 1)
    #     log_prob = dist.log_prob(raw_action).sum(dim=-1)
    #     return action.detach().numpy(), log_prob.detach()
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        action_mean = self.actor(state_tensor)
        action_std = torch.ones_like(action_mean) * 0.1  # Adjust as needed
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, min=-2.0, max=2.0)  # Clamp to (-1.5, 1.5)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), log_prob.detach().cpu()

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32, device='cuda'), torch.tensor(returns, dtype=torch.float32, device='cuda')

    def update(self, trajectories):
        states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32, device='cuda')
        actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.float32, device='cuda')
        old_log_probs = torch.tensor([t['log_prob'] for t in trajectories], dtype=torch.float32, device='cuda')
        rewards = [t['reward'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        values = [self.critic(torch.tensor(t['state'], dtype=torch.float32, device='cuda')).item() for t in trajectories]
        next_state = torch.tensor(trajectories[-1]['next_state'], dtype=torch.float32, device='cuda')
        next_value = self.critic(next_state).item()

        advantages, returns = self.compute_gae(rewards, dones, values, next_value)

        # Update actor
        action_means = self.actor(states)
        action_stds = torch.ones_like(action_means) * 0.1
        dist = torch.distributions.Normal(action_means, action_stds)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        values_tensor = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values_tensor, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
