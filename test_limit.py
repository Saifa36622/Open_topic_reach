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
from utility import feedback,reset,compute_reward,generate_cmd,compute_next_q,PPOAgent
import torch

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
    link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i])
    links.append(link)

# Initialize the UR3e robot model
ur3e = rtb.DHRobot(links, name='UR3e')
# ur3e.l
q = 6 * [0.0]

qd = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0])
dt = 0.01  # 50 milliseconds
q = np.zeros(6)  # Initial joint positions

backend = PyPlot()
backend.launch()
backend.add(ur3e)
fig = backend.fig
ax = fig.axes[0]

cmd = generate_cmd()
for point in cmd:
    ax.scatter(point[0], point[1], point[2], color='red', marker='o', s=50)

q_log = []
qd_log = []
qdd_log = []
ee_log = []
max_joint_acceleration = np.deg2rad(300)  # Convert 300°/s² to radians
qd_prev = np.zeros(6)
for i in range(500) :
    currrent_q, currrent_qqd, currrent_qqdd,ee_pose,valid = feedback(ur3e, dt)
    print(currrent_q, currrent_qqd, ee_pose)
    #  position control
    # q[0] += 0.01 


    # velo control
    q += qd * dt

    # for j in range (6) : 
    #     if j == 1 or j == 2:
    #         if (-np.pi < currrent_q[j] < np.pi) : 
    #             q[j] += qd[j] * dt
    #     else : 
    #         q[j] += qd[j] * dt

    # Compute the next joint positions and velocities with acceleration limiting
    # q, qd_prev = compute_next_q(q, qd, qd_prev, dt, max_joint_acceleration)
    ur3e.q = q
    backend.step(dt)
    
    q_log.append(currrent_q)
    qd_log.append(currrent_qqd)
    qdd_log.append(currrent_qqdd)
    ee_log.append(ee_pose)

ee_log = np.array(ee_log)  # Convert list to NumPy array for easy slicing
q_log_np = np.array(q_log)
qd_log_np = np.array(qd_log)
qdd_log_np = np.array(qdd_log)
time = np.arange(len(q_log_np)) * dt  # Time vector

# Plot joint positions
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, q_log_np[:, i], label=f'Joint {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Joint Position (rad)')
plt.title('Joint Positions Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot joint velocities
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, qd_log_np[:, i], label=f'Joint {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity (rad/s)')
plt.title('Joint Velocities Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot joint accelerations
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(time, qdd_log_np[:, i], label=f'Joint {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Joint Acceleration (rad/s²)')
plt.title('Joint Accelerations Over Time')
plt.legend()
plt.grid(True)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ee_log[:, 0], ee_log[:, 1], ee_log[:, 2], label='End-Effector Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('End-Effector Trajectory')
ax.legend()
plt.show()
