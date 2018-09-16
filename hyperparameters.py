# Hyper Parameters
import gym
import torch
from wrappers import *

'''Training Settings'''
USE_GPU = torch.cuda.is_available();print('USE GPU: '+str(USE_GPU))
EPISODE_NUM = 400
STEP_NUM = 2*10**7
BATCH_SIZE = 32
LR = 1e-4                   # learning rate
EPSILON = 0.0              # greedy policy
GAMMA = 0.99                 # reward discount

'''DQN Settings'''
STATE_LEN = 4
TARGET_REPLACE_ITER = 10**4   # target update frequency
MEMORY_CAPACITY = 10**6
MEMORY_MODE = 'PER'

'''Environment Settings'''
ENV_NAME = 'BreakoutNoFrameskip-v4'
env = wrap(gym.make(ENV_NAME))
test_env = test_wrap(gym.make(ENV_NAME))
# ENV_NAME = 'CartPole-v0'
# env = gym.make(ENV_NAME)
# test_env = gym.make(ENV_NAME)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape
USE_CNN = True

'''Save&Load Settings'''
SAVE = True
SAVE_FREQ = 10**4
LOAD = False
EVAL_PATH = './data/model/eval_net.pkl'
TARGET_PATH = './data/model/target_net.pkl'
RESULT_PATH = './data/plots/result.pkl'
RENDERING = False

'''Double&Dueling Settings'''
DOUBLE = False
DUEL = False
