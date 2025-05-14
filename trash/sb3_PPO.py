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

from math import pi
# Define action and observation space
ur_control.init('192.168.56.101')
# Send power on command
ur_control.power_on()

# Send break release command
ur_control.break_release()
# Link lengths (a)
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
print(ur3e.jacob0())
# class UR3eReachEnv(gym.Env):
#     """
#     Custom Environment for UR3e Reach Task compatible with OpenAI Gym interface.
#     """
#     metadata = {'render.modes': ['human']}

#     def __init__(self):
#         super(UR3eReachEnv, self).__init__()
        
#         # Example: 3D continuous action space for end-effector movements
#         self.action_space = spaces.Box(low=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), high=np.array([0.5, 0.5,0.5, 0.5, 0.5, 0.5,2.0,2.0]), shape=(8,), dtype=np.float32)
        
#         # Example: Observation space could include end-effector position and target position
#         self.observation_space = spaces.Box(low=np.array([-pi*2, -pi*2, -pi*2, -pi*2, -pi*2, -pi*2]), high=np.array([pi*2, pi*2, pi*2, pi*2, pi*2, pi*2]), shape=(6,), dtype=np.float32)
#         self.target_position = None
#         self.position = None

#     def reset(self,seed=None,**kwarg):
#         """
#         Reset the state of the environment to an initial state.
#         """
#         # Initialize the state
#         # Set the seed for random number generation
#         print("reset")
#         self.np_random, seed = gym.utils.seeding.np_random(seed)
        
#         # self.target_position = self.generate_multiple_targets(5)
#         self.target_position = self.generate_new_target()
#         p = [1.57, -math.pi/2, 0.0, 0.0, math.pi/2, 0.0]
#         check1 = ur_control.move_directly(p, a=1.4, v=1.05, t=0, r=0.02)
#         self.position = check1
#         print("done")
#         # time.sleep(1.0)
#         observation_array = np.array(self.position, dtype=np.float32)
#         print("end reset")
#         return observation_array
    
#     def generate_multiple_targets(self, n):
#         targets = []
#         for _ in range(n):
#             target = self.generate_new_target()
#             targets.append(target)
#         return np.array(targets) 

#     def generate_new_target(self):
#         x_range = (0.2, 0.5)
#         y_range = (-0.3, 0.3)
#         z_range = (0.1, 0.4)

#         x = self.np_random.uniform(*x_range)
#         y = self.np_random.uniform(*y_range)
#         z = self.np_random.uniform(*z_range)
#         return np.array([x, y, z], dtype=np.float32)

#     def check_done(self, observation):
        
#         end_effector_position = ur3e.fkine(observation).t  
#         distance = np.linalg.norm(end_effector_position - self.target_position)
        
        
#         reach_threshold = 0.05

#         done_threshold = 5.0

#         if distance < reach_threshold:
#             return 2,True  
#         elif distance > done_threshold :
#             return -2,True  
        
#         return 0,False  
        

#     def cal_reward(self,observation):

#         joint_angles = np.array(observation, dtype=np.float32)

#         # Compute the end-effector's pose using forward kinematics
#         end_effector_pose = ur3e.fkine(joint_angles)

#         # Extract the position component from the pose
#         end_effector_position = end_effector_pose.t  # This gives the translation vector

#         # Initialize a high reward value (to minimize the distance)
#         # min_distance = float('inf')
#         # target_reached = False
#         # self.reach_threshold = 0.05

#         # # Iterate through all the target positions to find the closest one
#         # for target_position in self.target_position:
#         #     # Calculate the Euclidean distance between the current end-effector position and the target
#         #     distance = np.linalg.norm(end_effector_position - target_position)

#         #     # If the agent is very close to a target (within a threshold), give a bonus reward and mark target as reached
#         #     if distance < self.reach_threshold:
#         #         target_reached = True
#         #         reward = 1
                
#         #     elif distance < min_distance:
#         #         # Track the closest target
#         #         min_distance = distance

#         # if not target_reached:
#         #     # If no target is reached, the reward is the negative of the minimum distance
#         #     reward = -min_distance

#         # return reward
#         # Calculate the Euclidean distance between the current and target positions
#         distance = np.linalg.norm(end_effector_position - self.target_position)

#         # Define the reward as the negative distance (closer to target yields higher reward)
#         reward = -distance

#         return reward

#     def step(self, action):
#         """
#         Execute one time step within the environment.
#         """
#         print("step")
#         # Apply action to the environment
#         # Compute reward, next state, and done flag
#         # return observation, reward, done, info
#         for i in range (6) :
#             self.position[i] += action[i]
#         self.position = ur_control.move_directly(self.position[0:5], a=action[6], v=action[7], t=0, r=0.02)
#         reward,done = self.check_done(self.position)
#         if not done :
#             reward = self.cal_reward(self.position)

#         info = {}
        
#         observation_array = np.array(self.position, dtype=np.float32)
#         return observation_array, reward, done, info

#     def render(self, mode='human'):
#         pass

#     def close(self):
#         pass


# # Initialize environment and check its correctness
# env = UR3eReachEnv()
# # check_env(env)  # Optional: validate your environment

# # Choose the RL algorithm: PPO or SAC
# model = PPO("MlpPolicy", env, verbose=1)

# # Train the model
# model.learn(total_timesteps=50000)  # You can adjust total_timesteps for longer training

# # Save the model after training
# model.save("ur3e_reach_model")

# # Optionally: Evaluate the model
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()