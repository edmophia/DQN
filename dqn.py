import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from NN import *
from hyperparameters import *
from replay_memory import ReplayBuffer, PrioritizedReplayBuffer
from collections import deque
import random

class DQN(object):
    def __init__(self):
        if USE_CNN:
            if USE_GPU:
                self.eval_net, self.target_net = ConvNet().cuda(), ConvNet().cuda()
            else:
                self.eval_net, self.target_net = ConvNet(), ConvNet()
        else:
            if USE_GPU:
                self.eval_net, self.target_net = Net().cuda(), Net().cuda()
            else:
                self.eval_net, self.target_net = Net(), Net()


        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0
        
        # Create the replay buffer
        if MEMORY_MODE == 'PER':
            self.replay_buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY, alpha=PER_ALPHA)
        else:
            self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
            
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    def choose_action(self, x, EPSILON):
        if USE_GPU:
            x = Variable(torch.FloatTensor(x)).cuda()
        else:
            x = Variable(torch.FloatTensor(x))

        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x.unsqueeze(0))
            if USE_GPU:
                action = torch.argmax(actions_value).data.cpu().numpy() # return the argmax
            else:
                action = torch.argmax(actions_value).data.numpy()     # return the argmax;
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self, beta):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
         # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if MEMORY_MODE == 'PER':
            experience = self.replay_buffer.sample(BATCH_SIZE, beta=beta)
            (b_state_memory, b_action_memory, b_reward_memory, b_next_state_memory, b_done, b_weights, b_idxes) = experience
        else:
            b_state_memory, b_action_memory, b_reward_memory, b_next_state_memory, b_done = self.replay_buffer.sample(BATCH_SIZE)
            b_weights, b_idxes = np.ones_like(b_reward_memory), None

        if USE_GPU:
            b_s = Variable(torch.FloatTensor(b_state_memory)).cuda()
            b_a = Variable(torch.LongTensor(b_action_memory)).cuda()
            b_r = Variable(torch.FloatTensor(b_reward_memory)).cuda()
            b_s_ = Variable(torch.FloatTensor(b_next_state_memory)).cuda()
            b_d = Variable(torch.FloatTensor(b_done)).cuda()
        else:
            b_s = Variable(torch.FloatTensor(b_state_memory))
            b_a = Variable(torch.LongTensor(b_action_memory))
            b_r = Variable(torch.FloatTensor(b_reward_memory))
            b_s_ = Variable(torch.FloatTensor(b_next_state_memory))
            b_d = Variable(torch.FloatTensor(b_done))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1)).view(-1)  # shape (batch, 1)

        if DOUBLE:
            _ , best_actions = self.eval_net.forward(b_s_).detach().max(1)
            q_next = self.target_net(b_s_).detach()    # detach from graph, don't backpropagate
            q_target = b_r + GAMMA *(1.-b_d)* q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # shape (batch, 1)
        else:
            q_next = self.target_net(b_s_).detach()    # detach from graph, don't backpropagate
            q_target = b_r + GAMMA *(1.-b_d)* q_next.max(1)[0]  # shape (batch, 1)
            
        loss = F.smooth_l1_loss(q_eval, q_target, reduce= False)
        loss = torch.mean(torch.Tensor(b_weights).cuda()*loss)
        td_error = (q_target - q_eval).data.cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(),10.)
        self.optimizer.step()
        
        if MEMORY_MODE == 'PER':
            new_priorities = np.abs(td_error) + PER_EPSILON
            self.replay_buffer.update_priorities(b_idxes, new_priorities)

    def save_model(self):
        # save evaluation network and target network simultaneously
        self.eval_net.save(EVAL_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load evaluation network and target network simultaneously
        self.eval_net.load(EVAL_PATH)
        self.target_net.load(TARGET_PATH)