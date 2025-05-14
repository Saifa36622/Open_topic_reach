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
from utility import feedback,reset,compute_reward,compute_next_q,PPOAgent
import torch
import wandb

# Link lengths (a)
a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]

# Link twists (alpha)
alpha = [pi/2, 0.0, 0.0, pi/2, -pi/2, 0.0]

# Link offsets (d)
d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]

qlim = np.array([
    [-2*np.pi, 2*np.pi],  # Joint 1
    [np.pi, np.pi],  # Joint 2
    [np.pi, np.pi],  # Joint 3
    [-2*np.pi, 2*np.pi],  # Joint 4
    [-2*np.pi, 2*np.pi],  # Joint 5
    [-2*np.pi, 2*np.pi]   # Joint 6
])
# Create the list of links
links = []
for i in range(6):
    link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i],qlim=qlim[i])
    links.append(link)

# Initialize the UR3e robot model
ur3e = rtb.DHRobot(links, name='UR3e')
# ur3e.l
# q = 6 * [0.0]
# ur3e.plot(q)

# dir = "output"
# os.makedirs(dir, exist_ok=True)
# file_path = os.path.join(dir, 'ur3e_plot.png')


# q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# qd = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

dt = 0.01  # 50 milliseconds

q = np.zeros(6)  # Initial joint positions
qd_prev = np.zeros(6)

epoch = 1000
timestep = 1000
backend = PyPlot()
backend.launch()
backend.add(ur3e)
fig = backend.fig
ax = fig.axes[0]
max_joint_acceleration = np.deg2rad(300)  # Convert 300°/s² to radians

agent = PPOAgent(
    state_dim=37,
    action_dim=6,
    actor_lr= 3e-4,
    critic_lr= 0.001,
    gamma=0.99,
    eps_clip=0.2
    )
# -------------------------------------------------------------------------------------
# train loop
# Observation space -> [timestep,position[6],velo[6],acc[6],end_pose[3],command[5]*3 = 15] -> 37

currrent_q, currrent_qqd, currrent_qqdd,end_pos ,valid = feedback(ur3e, dt)

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

wandb.init(project="Open_topic",name="test")

run = 1

for i in range (epoch) :

    # wandb.log({
    #     'episode' : i
    # })
    reward_list = []
    done_list = []

    actor_loss_sum = 0
    critic_loss_sum = 0
    reward_sum = 0
    log_prob_sum = 0
    done = False
    reward = 0
    qd_action = 6 * [0.0]
    log_prob = 0
    distance_sum = 0

    backend.close()

    backend.launch()

    ur3e.q = 6 * [0.0]

    backend.add(ur3e)
    fig = backend.fig
    ax = fig.axes[0]
    

    state,cmd = reset(ur3e,dt)

    


    print("reset")

    
    backend.step(dt)

    currrent_q, currrent_qqd, currrent_qqdd,end_pos ,valid = feedback(ur3e, dt)
    next_state = np.concatenate(([2],currrent_q,currrent_qqd ,currrent_qqdd,end_pos, cmd.flatten()))


    trajectories = []

    trajectories.append({
        'state': state,
        'action': qd_action,
        'log_prob': log_prob,
        'reward': reward ,
        'done': done,
        'next_state' : next_state
    })

    # # Update the agent
    actor_loss,critic_loss = agent.update(trajectories)
    # ax.clear()
    
    # for point in cmd:
    #     ax.scatter(point[0], point[1], point[2], color='red', marker='o', s=50)

    if 'scatter_points' in locals():
        for p in scatter_points:
            p.remove()

    scatter_points = []
    for point in cmd:
        scatter = ax.scatter(point[0], point[1], point[2], color='red', marker='o', s=50)
        scatter_points.append(scatter)

    # time.sleep(1)
    for j in range (1,timestep+1) :

        # selected action 

        qd_action, log_prob = agent.select_action(state)

        # do to env 
        #  ------------------------------------------------------------------------------------

        #  position control
        # q[0] += 0.01 


        # velo control
        # q += action * dt


        q, qd_prev = compute_next_q(q, qd_action, qd_prev, dt, max_joint_acceleration)

        ur3e.q = q
    
        backend.step(dt)

        #  ------------------------------------------------------------------------------------

        # feedback
        currrent_q, currrent_qqd, currrent_qqdd, end_pos, valid = feedback(ur3e, dt)

        if not valid:
            continue  # Skip logging the first invalid step

        # currrent_q, currrent_qqd, currrent_qqdd ,end_pos = feedback(ur3e, dt)
        # print(currrent_q, currrent_qqd, currrent_qqdd)


        reward , done ,cmd,distance,n_success = compute_reward(ur3e,cmd,j,dt)
        # print(currrent_q, currrent_qqd, currrent_qqdd,end_pos, cmd.flatten())
        next_state = np.concatenate(([j+1],currrent_q,currrent_qqd ,currrent_qqdd,end_pos, cmd.flatten()))

        next_value = agent.critic(torch.tensor(next_state, dtype=torch.float32, device='cuda')).item() if not done else 0
        reward_list.append(reward)
        done_list.append(done)
        # returns = agent.compute_returns(reward_list , done_list , next_value)

        trajectories = []

        trajectories.append({
            'state': state,
            'action': qd_action,
            'log_prob': log_prob,
            'reward': reward ,
            'done': done,
            'next_state' : next_state
        })

        # # Update the agent
        actor_loss,critic_loss =agent.update(trajectories)

        state = next_state

        actor_loss_sum += actor_loss
        critic_loss_sum += critic_loss
        reward_sum += reward
        log_prob_sum += log_prob
        distance_sum += distance
     
        if done :
            break

    wandb.log({
    'episode' : i+1,
    "reward" : reward_sum / 1000,
    "actor loss" : actor_loss_sum / 1000,
    "critic_loss" : critic_loss_sum /1000,
    "log_prob" : log_prob_sum /1000,
    "distance" : distance_sum/1000,
    "number of success" : n_success
    })

    torch.save(agent.actor.state_dict(), os.path.join(save_dir,f"run_{run}", f"actor_episode_{i+1}.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir,f"run_{run}", f"critic_episode_{i+1}.pth"))

wandb.finish()

# --------------------------------------------------------------------------------------

