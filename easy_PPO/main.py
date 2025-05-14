"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
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
        
        # Example: 3D continuous action space for end-effector movements
        self.action_space = spaces.Box(low=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), high=np.array([0.5, 0.5,0.5, 0.5, 0.5, 0.5,2.0,2.0]), shape=(8,), dtype=np.float32)
        
        # Example: Observation space could include end-effector position and target position
        self.observation_space = spaces.Box(low=np.array([-pi*2, -pi*2, -pi*2, -pi*2, -pi*2, -pi*2]), high=np.array([pi*2, pi*2, pi*2, pi*2, pi*2, pi*2]), shape=(6,), dtype=np.float32)
        self.target_position = None
        self.position = None

    def reset(self,seed=None,**kwarg):
        """
        Reset the state of the environment to an initial state.
        """
        # Initialize the state
        # Set the seed for random number generation
        print("reset")
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # self.target_position = self.generate_multiple_targets(5)
        self.target_position = self.generate_new_target()
        p = [1.57, -math.pi/2, 0.0, 0.0, math.pi/2, 0.0]
        check1 = ur_control.move_directly(p, a=1.4, v=1.05, t=0, r=0.02)
        self.position = check1
        print("done")
        # time.sleep(1.0)
        observation_array = np.array(self.position, dtype=np.float32)
        print("end reset")
        return observation_array
    
    def generate_multiple_targets(self, n):
        targets = []
        for _ in range(n):
            target = self.generate_new_target()
            targets.append(target)
        return np.array(targets) 

    def generate_new_target(self):
        x_range = (0.2, 0.5)
        y_range = (-0.3, 0.3)
        z_range = (0.1, 0.4)

        x = self.np_random.uniform(*x_range)
        y = self.np_random.uniform(*y_range)
        z = self.np_random.uniform(*z_range)
        return np.array([x, y, z], dtype=np.float32)

    def check_done(self, observation):
        
        end_effector_position = ur3e.fkine(observation).t  
        distance = np.linalg.norm(end_effector_position - self.target_position)
        
        
        reach_threshold = 0.05

        done_threshold = 5.0

        if distance < reach_threshold:
            return 2,True  
        elif distance > done_threshold :
            return -2,True  
        
        return 0,False  
        

    def cal_reward(self,observation):

        joint_angles = np.array(observation, dtype=np.float32)

        # Compute the end-effector's pose using forward kinematics
        end_effector_pose = ur3e.fkine(joint_angles)

        # Extract the position component from the pose
        end_effector_position = end_effector_pose.t  # This gives the translation vector

        # Initialize a high reward value (to minimize the distance)
        # min_distance = float('inf')
        # target_reached = False
        # self.reach_threshold = 0.05

        # # Iterate through all the target positions to find the closest one
        # for target_position in self.target_position:
        #     # Calculate the Euclidean distance between the current end-effector position and the target
        #     distance = np.linalg.norm(end_effector_position - target_position)

        #     # If the agent is very close to a target (within a threshold), give a bonus reward and mark target as reached
        #     if distance < self.reach_threshold:
        #         target_reached = True
        #         reward = 1
                
        #     elif distance < min_distance:
        #         # Track the closest target
        #         min_distance = distance

        # if not target_reached:
        #     # If no target is reached, the reward is the negative of the minimum distance
        #     reward = -min_distance

        # return reward
        # Calculate the Euclidean distance between the current and target positions
        distance = np.linalg.norm(end_effector_position - self.target_position)

        # Define the reward as the negative distance (closer to target yields higher reward)
        reward = -distance

        return reward

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        print("step")
        # Apply action to the environment
        # Compute reward, next state, and done flag
        # return observation, reward, done, info
        for i in range (6) :
            self.position[i] += action[i]
        self.position = ur_control.move_directly(self.position[0:5], a=action[6], v=action[7], t=0, r=0.02)
        reward,done = self.check_done(self.position)
        if not done :
            reward = self.cal_reward(self.position)

        info = {}
        
        observation_array = np.array(self.position, dtype=np.float32)
        return observation_array, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.


	env = gym.make(UR3eReachEnv(), render_mode='human' if args.mode == 'test' else 'rgb_array')

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
