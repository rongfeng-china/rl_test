from RL_brain import DeepQNetwork
from arm_env import ArmEnv
import numpy as np
import os
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

RL = DeepQNetwork(n_actions = ACTION_DIM, 
            n_features = STATE_DIM,
            learning_rate = 0.01, e_greedy = 0.9,
            replace_target_iter = 100, memory_size = 2000,
            e_greedy_increment = .0008)

total_steps = 0

for i_episode in range(10000):
    ## initialization
    observation = env.reset()
    ep_r = 0

    while True:
        env.render()
        action = RL.choose_action(observation)
        #print (action)
        observation_, r, done= env.step(action)

        
        ## store in memory    
        RL.store_transition(observation,action,r,observation_)
 
        # reward for espisode
        ep_r += r

        if total_steps > 500:
            RL.learn()

        if done:
            #print 'i_episode: '+str(i_episode)
            #print 'ep_r: %f' %(round(ep_r,2))
            #print 'epsilon: %f' %(round(RL.epsilon,2))

            break
       
        observation = observation_
        total_steps += 1

RL.plot_cost()
