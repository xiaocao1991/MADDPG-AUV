# main function that sets up environments
# perform training loop

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
LR_ACTOR    =   1e-3     # Learning rate of the actor
LR_CRITIC   =   1e-4     # Learning rate of the critic
WEIGHT_DECAY =  0#1e-5     # L2 weight decay
UPDATE_EVERY =  30       # How many steps to take before updating target networks
UPDATE_TIMES =  20       # Number of times we update the networks
SEED = 3                 # Seed for random numbers
BENCHMARK   =   False
EXP_REP_BUF =   False     # Experienced replay buffer activation
PRE_TRAINED =   False    # Use a previouse trained network as imput weights
#Scenario used to train the networks
SCENARIO    =   "simple_spread_ivan" 
# SCENARIO    =   "simple_track_ivan" 
RENDER = False #in BSC machines the render doesn't work

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    seeding(seed = SEED)
    # number of parallel agents
    parallel_envs = 6
    # number of agents per environment
    num_agents = 6
    # number of landmarks (or targets) per environment
    num_landmarks = 6
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 400000
    episode_length = 35
    # how many episodes to save policy and gif
    save_interval = 1000
    t = 0
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 0.05 #was 2, try 0.5
    noise_reduction = 0.999

    # how many episodes before update
    # episode_per_update = UPDATE_EVERY * parallel_envs
    common_folder = time.strftime(r"\%m%d%y_%H%M%S")
    log_path = os.getcwd()+common_folder+r"\log"
    model_dir= os.getcwd()+common_folder+r"\model_dir"
    
    os.makedirs(model_dir, exist_ok=True)
    
    if BENCHMARK:
        benchmark_dir = os.getcwd()+common_folder+r"\benchmark_dir"
        os.makedirs(benchmark_dir, exist_ok=True) 
    
    # initialize environment
    torch.set_num_threads(parallel_envs)
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, benchmark = BENCHMARK)
       
    # initialize replay buffer
    if EXP_REP_BUF == False:
        buffer = ReplayBuffer(int(BUFFER_SIZE))
    else:
        buffer = ReplayBuffer_SummTree(int(BUFFER_SIZE), SEED) #Experienced replay buffer
        priority = np.ones(num_agents) #initial experienced replay buffer priority
    
    # initialize policy and critic
    maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY)
    logger = SummaryWriter(log_dir=log_path)
    
    agents_reward = []
    for n in range(num_agents):
        agents_reward.append([])
    # agent0_reward = []
    # agent1_reward = []
    # agent2_reward = []
    
    if BENCHMARK:
        agent_info = []
        for i in range(num_agents):
            agent_info.append((0,0,0,0))  # placeholder for benchmarking info
    
    # training loop
    # show progressbar
    #import progressbar as pb
    #widget = ['\repisode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
    #          pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    #timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()
    
    import tqdm
    #initializing progress bar object
    timer_bar = tqdm.tqdm(range(number_of_episodes),desc='Episode',position=0)
    
    
    
    if PRE_TRAINED == True:
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032621_224252\model_dir\episode-99000.pt' #test1 6 agents
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032821_102717\model_dir\episode-99000.pt' #test2 6 agents + pretrined from previous
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\032921_160324\model_dir\episode-99000.pt' #test3 6 agents pre-pretrined
        # trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\033021_203450\model_dir\episode-98004.pt' #test3 6 agents pre-pre-pretrined
        trained_checkpoint = r'E:\Ivan\UPC\UDACITY\DRL_Nanodegree\Part4\MADDPG\040921_222255\model_dir\episode-299994.pt' #Test with slighly reward function modified 300.000 iteration
    
        aux = torch.load(trained_checkpoint)
        for i in range(num_agents):  
            # load the weights from file
            maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
            maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
    
    print('Starting iterations...')
    for episode in range(0, number_of_episodes, parallel_envs):

        #timer.update(episode)
        timer_bar.update()

        #Reset the environment
        all_obs = env.reset() 
        #Reset the noise
        for i in range(num_agents):
            maddpg.maddpg_agent[i].noise.reset()
        #Reset the rewards
        reward_this_episode = np.zeros((parallel_envs, num_agents))  
        
        # flip the first two indices
        obs_roll = np.rollaxis(all_obs,1)
        obs = transpose_list(obs_roll)
        
        # save info or not
        save_info = ((episode) % save_interval < parallel_envs or episode==number_of_episodes-parallel_envs)
        frames = []
        tmax = 0
        
        
        if save_info == True and RENDER == True:
            frames.append(env.render('rgb_array'))

        for episode_t in range(episode_length):
            
            # get actions
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            
    
            actions_array = torch.stack(actions).detach().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array,1)
            
            # environment step
            # step forward one frame
            # next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
            next_obs, rewards, dones, info = env.step(actions_for_env)
            # rewards_sum += np.mean(rewards)
            
            # collect experience
            # add data to buffer
            # transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
            transition = (obs, actions_for_env, rewards, next_obs, dones)
            if EXP_REP_BUF == False:
                buffer.push(transition)
            else:
                buffer.push(transition,priority)                
            reward_this_episode += rewards

            # obs, obs_full = next_obs, next_obs_full
            obs = next_obs
            
            # increment global step counter
            t += parallel_envs
            
            # save gif frame
            if save_info == True and RENDER == True:
                frames.append(env.render('rgb_array'))
                tmax+=1
                
            # for benchmarking learned policies
            if BENCHMARK:
                for e, inf in enumerate(info):
                    for a in range(num_agents):
                        agent_info[a] = np.add(agent_info[a],(inf['n'][a]))
            
            # finish the episode if done
            if dones.any():
                break
            
        #Reduce the quantity of noise added to the action
        noise *= noise_reduction
                
        # update once after every episode_per_update 
        # if len(buffer) > BATCH_SIZE and episode % episode_per_update < parallel_envs:
        if len(buffer) > BATCH_SIZE and episode % UPDATE_EVERY < parallel_envs:
            for _ in range(UPDATE_TIMES):
                priority = np.zeros(num_agents)
                for a_i in range(num_agents):
                    if EXP_REP_BUF == False:
                        samples = buffer.sample(BATCH_SIZE)
                        priority = maddpg.update(samples, a_i, logger)
                    else:
                        samples, indexes = buffer.sample(BATCH_SIZE)
                        new_priorities = maddpg.update(samples, a_i, logger)
                        priority[a_i] = buffer.update(indexes, new_priorities)
                if EXP_REP_BUF == True:
                    priority /= num_agents
            maddpg.update_targets() #soft update the target network towards the actual networks

                
        for i in range(parallel_envs):
            for n in range(num_agents):
                agents_reward[n].append(reward_this_episode[i,n])
            # agent0_reward.append(reward_this_episode[i,0])
            # agent1_reward.append(reward_this_episode[i,1])
            # agent2_reward.append(reward_this_episode[i,2])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            # avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward), np.mean(agent2_reward)]
            avg_rewards = []
            for n in range(num_agents):
                avg_rewards.append(np.mean(agents_reward[n])) 
            # agent0_reward = []
            # agent1_reward = []
            # agent2_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        #saving model
        save_dict_list =[]
        if save_info:
            
            for i in range(num_agents):

                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
            
            if RENDER == True:
                # save gif files
                imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                                frames, duration=.04)
            
            #save benchmark
            if BENCHMARK:
                file1 = open(benchmark_dir+r"\episode-{}.txt".format(episode),"w")#append mode 
                file1.write(str(np.array(agent_info)/t)) 
                file1.close() 

    env.close()
    logger.close()
    #timer.finish()

if __name__=='__main__':
    main()
    
    

