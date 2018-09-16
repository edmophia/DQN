from hyperparameters import *
from dqn import DQN
import os
import pickle
from copy import deepcopy
import numpy as np
import time

dqn = DQN()

#model load with checking existence of file
if LOAD and os.path.isfile(EVAL_PATH) and os.path.isfile(TARGET_PATH) and os.path.isfile(RESULT_PATH):
    dqn.load_model()

    pkl_file = open(RESULT_PATH,'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()

    print('Load complete!')
else:
    result = []
    print('Initialize results!')

print('\nCollecting experience...')

epi_step = 0
entire_ep_r = 0.
entire_ep_rs = []
start_time = time.time()

while dqn.memory_counter <= STEP_NUM:
    s = env.reset()
    if USE_CNN:
        s = np.reshape([np.array(s)],(STATE_LEN,84,84))
    else:
        s.reshape(1,-1)

    # episode's reward
    ep_r = 0.

    while True:
        a = dqn.choose_action(s, EPSILON)

        # take action
        s_, r, done, info = env.step(a)
        ep_r += r
        clip_r = np.sign(r)

        # reshape the state
        if USE_CNN:
            s_ = np.reshape([np.array(s_)],(STATE_LEN,84,84))
        else:
            s_.reshape(1,-1)

        # store the transition
        dqn.store_transition(s, a, clip_r, s_, float(done))

        # annealing the epsilon, beta
        if dqn.memory_counter <= 1e+6:
            EPSILON += 0.9/(1e+6)
        elif dqn.memory_counter <= (2e+7):
            EPSILON += 0.09/(2e+7 - 1e+6)
        PER_BETA += (1 - PER_BETA_INIT)/(2e+7)

        if (5e+4 <= dqn.memory_counter) and (dqn.memory_counter % 4 == 0):
            dqn.learn(PER_BETA)

        if dqn.memory_counter % SAVE_FREQ == 0:
            #print return
            end_time = time.time()
            mean_100_ep_reward = round(np.mean(entire_ep_rs[-101:-1]),2)
            result.append(mean_100_ep_reward)
            print('Ep: ',epi_step,
                  '| Mean ep 100 return: ', mean_100_ep_reward, '/Used Time:',round(end_time-start_time,2)
                 ,'/Used Step:',dqn.memory_counter)

            if SAVE:
                #save model
                dqn.save_model()
                pkl_file = open(RESULT_PATH, 'wb')
                pickle.dump(np.array(result), pkl_file)
                pkl_file.close()
                print('Save complete!')

        if done:
            entire_ep_r += ep_r
            epi_step += 1
            if epi_step % 5 == 0:
                entire_ep_rs.append(entire_ep_r)
                entire_ep_r = 0.
            break

        s = s_

        if RENDERING:
            env.render()
