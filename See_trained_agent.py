# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:37:13 2021

@author: Usuari

5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
import envs
from buffer import ReplayBuffer, ReplayBuffer_SummTree
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor
import time

# for saving gif
import imageio

BUFFER_SIZE =   int(1e6) # Replay buffer size
BATCH_SIZE  =   512      # Mini batch size
GAMMA       =   0.95     # Discount factor
TAU         =   0.01     # For soft update of target parameters 
LR_ACTOR    =   1e-2     # Learning rate of the actor
LR_CRITIC   =   1e-3     # Learning rate of the critic
WEIGHT_DECAY =  0 #1e-5     # L2 weight decay
UPDATE_EVERY =  30       # How many steps to take before updating target networks
UPDATE_TIMES =  20       # Number of times we update the networks
SEED = 3534                 # Seed for random numbers
BENCHMARK   =   False
EXP_REP_BUF =   False     # Experienced replay buffer activation
PRE_TRAINED =   True    # Use a previouse trained network as imput weights
#Scenario used to train the networks
SCENARIO    =   "simple_track_ivan" 
RENDER = True #in BSC machines the render doesn't work
PROGRESS_BAR = True #if we want to render the progress bar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding(seed = SEED)
    # number of parallel agents
    parallel_envs = 1
    # number of agents per environment
    num_agents = 1
    # number of landmarks (or targets) per environment
    num_landmarks = 1
    
    # initialize environment
    torch.set_num_threads(parallel_envs)
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, num_landmarks=num_landmarks, benchmark = BENCHMARK)
       
    # initialize policy and critic
    maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE)
    agents_reward = []
    for n in range(num_agents):
        agents_reward.append([])
    
    if PRE_TRAINED == True:
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032521_163018\model_dir\episode-59994.pt' #test1 2 agents
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032521_211315\model_dir\episode-59994.pt' #test1 2 agents
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032621_054252\model_dir\episode-36000.pt' #test1 2 agents
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032821_102717\model_dir\episode-99000.pt' #test1 6 agents
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032921_160324\model_dir\episode-99000.pt' #test2 6 agents pretrined
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\033021_203450\model_dir\episode-73002.pt' #test2 6 agents pretrined
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\033121_232315\model_dir\episode-265002.pt' #test2 6 agents 3 layers NN
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\040521_000716\model_dir\episode-111000.pt' #test1 6 agents new reward function
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\040621_143510\model_dir\episode-153000.pt' #test1 6 agents new new reward function
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\040921_222255\model_dir\episode-299994.pt' #test1 6 agents new new reward function new positive reward
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\041321_204450\model_dir\episode-196002.pt' #test1 6 agents new new reward function new positive reward and pretrined 
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\051021_140623\model_dir\episode-107000.pt' #first test with PF with one agent and one landmark
        
        #New tests with PF and LS simple_track_ivan.py 
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\061721_144815\model_dir\episode-399992.pt' #first test with LS with one agent and one landmark (episode_length=35) This works better, it has learned to stay close to the landmark and make small movements to maintain the error.
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\061721_222642\model_dir\episode-399006.pt' #second test with LS with one agent and one landmark (episode_length=60) This works a little worst than the previouse, it has a similar behaviour, but it moves lower, and therefore, the error is greater.
        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\061821_105753\model_dir\episode-399992.pt' #third test with LS with one agent and one landmark (episode_length=35) In this case, the observation state is the estimated landmark position instead of the true landmark position as the two previous tests. It works prety well
        
        
        aux = torch.load(trained_checkpoint)
        for i in range(num_agents):  
            # load the weights from file
            maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
            maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
    
    #Reset the environment
    all_obs = env.reset() 
    # flip the first two indices
    obs_roll = np.rollaxis(all_obs,1)
    obs = transpose_list(obs_roll)
    
    scores = 0                
    t = 0
    while True:
        env.render('rgb_array')
        t +=1
        # select an action
        actions = maddpg.act(transpose_to_tensor(obs), noise=0.)                
        actions_array = torch.stack(actions).detach().numpy()
        actions_for_env = np.rollaxis(actions_array,1)
        # send all actions to the environment
        next_obs, rewards, dones, info = env.step(actions_for_env)
        print('RMSE = %.3f'%rewards[0])
        # update the score (for each agent)
        scores += np.sum(rewards)            
        # print ('\r\n Rewards at step %i = %.3f'%(t,scores))
        # roll over states to next time step                    
        obs = next_obs                              
        # print("Score: {}".format(scores))
        if np.any(dones):
            print('done')
            print('Next:')
    env.close()
    
if __name__=='__main__':
    main()
    