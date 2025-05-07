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


class UR3eReachEnv(gym.Env):
    """
    Custom Environment for UR3e Reach Task compatible with OpenAI Gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UR3eReachEnv, self).__init__()
        
        # Define action and observation space
        # Example: 3D continuous action space for end-effector movements
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Example: Observation space could include end-effector position and target position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.target_position = None  

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # Initialize the state
        self.target_position = self.generate_new_target()
        p2 = [0.0, -math.pi/2, math.pi/2, 0.0, math.pi/2, 0.0]
        observation = ur_control.move_directly(p2, a=1.4, v=1.05, t=0, r=0) 
        time.sleep(1.0)
        observation_array = np.array(observation, dtype=np.float32)
        return observation_array
    
    def generate_new_target(self):
        x_range = (0.2, 0.5)
        y_range = (-0.3, 0.3)
        z_range = (0.1, 0.4)

        x = self.np_random.uniform(*x_range)
        y = self.np_random.uniform(*y_range)
        z = self.np_random.uniform(*z_range)
        return np.array([x, y, z], dtype=np.float32)
    
    def check_done(self,observation):
        pass

    def cal_reward(self,observation):

        joint_angles = np.array(observation, dtype=np.float32)

        # Compute the end-effector's pose using forward kinematics
        end_effector_pose = ur3e.fkine(joint_angles)

        # Extract the position component from the pose
        end_effector_position = end_effector_pose.t  # This gives the translation vector

        # Calculate the Euclidean distance between the current and target positions
        distance = np.linalg.norm(end_effector_position - self.target_position)

        # Define the reward as the negative distance (closer to target yields higher reward)
        reward = -distance

        return reward



    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Apply action to the environment
        # Compute reward, next state, and done flag
        # return observation, reward, done, info

        observation = ur_control.move_directly(action, a=1.4, v=1.05, t=0, r=0)
        reward,done = self.check_done(observation)
        if not done :
            reward = self.cal_reward(observation)

        info = {}
        
        observation_array = np.array(observation, dtype=np.float32)
        return observation_array, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


env = UR3eReachEnv()
check_env(env)  # Optional: validate your environment

model = PPO("MlpPolicy", env, verbose=1)
# or
# model = SAC("MlpPolicy", env, verbose=1)