# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax
import numpy as np



class MADDPG:
    def __init__(self, num_agents = 3, num_landmarks = 1, discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu'):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        # (in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        # self.maddpg_agent = [DDPGAgent(14, 16, 8, 2, 20, 32, 16), 
        #                      DDPGAgent(14, 16, 8, 2, 20, 32, 16), 
        #                      DDPGAgent(14, 16, 8, 2, 20, 32, 16)]
    
        # self.maddpg_agent = [DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay), 
        #                      DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay), 
        #                      DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay)]
        #layers configuration
        # in_actor = num_landmarks*2 + (num_agents-1)*2 + 2+2 #x-y of landmarks + x-y of others + x-y and x-y velocity of current agent
        in_actor = num_landmarks*2 + (num_agents-1)*2 + 2+2 + num_landmarks*num_agents #x-y of landmarks + x-y of others + x-y and x-y velocity of current agent + range to landmarks
        hidden_in_actor = in_actor*15
        hidden_out_actor = int(hidden_in_actor/2)
        out_actor = 2 #each agent have 2 continuous actions on x-y plane
        in_critic = in_actor * num_agents # the critic input is all agents concatenated
        hidden_in_critic = in_critic * 4 + out_actor * num_agents
        hidden_out_critic = int(hidden_in_critic/2)
        #RNN
        rnn_num_layers = 2 #two stacked RNN to improve the performance (default = 1)
        rnn_hidden_size_actor = hidden_in_actor
        rnn_hidden_size_critic = hidden_in_critic - out_actor * num_agents
        
        
        
        print('Actor NN configuration:')
        print('Input nodes number:            ',in_actor)
        print('Hidden 1st layer nodes number: ',hidden_in_actor)
        print('Hidden 2nd layer nodes number: ',hidden_out_actor)
        print('Output nodes number:           ',out_actor)
        print('Critic NN configuration:')
        print('Input nodes number:            ',in_critic)
        print('Hidden 1st layer nodes number: ',hidden_in_critic)
        print('Hidden 2nd layer nodes number: ',hidden_out_critic)
        print('Output nodes number:           ',out_actor)
        
        self.maddpg_agent = [DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        # self.maddpg_agent = [DDPGAgent(14, 128, 128, 2, 48, 128, 128, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        
        #initial priority for the experienced replay buffer
        self.priority = 1.
        
        #device 'cuda' or 'cpu'
        self.device = device

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions_next

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions_next = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions_next

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents 
            Update parameters of agent model based on sample from replay buffer
            Inputs:
                samples: tuple of (observations, full observations, actions, rewards, next
                        observations, full next observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
                agent_number (int): index of agent to update
                logger (SummaryWriter from Tensorboard-Pytorch):
                    If passed in, important quantities will be logged
        """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        
        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)
        
        obs_full = torch.cat(obs, dim=1)
        next_obs_full = torch.cat(next_obs, dim=1)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions_next = self.target_act(next_obs) 
        target_actions_next = torch.cat(target_actions_next, dim=1)
        # target_critic_input = torch.cat((next_obs_full.t(),target_actions_next), dim=1).to(self.device)
        # target_critic_input = torch.cat((next_obs_full,target_actions_next), dim=1).to(self.device)
        target_critic_input_1 = next_obs_full.to(self.device)
        target_critic_input_2 = target_actions_next.to(self.device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input_1, target_critic_input_2)
        
        # Compute Q targets (y) for current states (y_i)
        y = reward[agent_number].view(-1, 1).to(self.device) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1)).to(self.device)

        # Compute Q expected (q) 
        action = torch.cat(action, dim=1)
        # critic_input = torch.cat((obs_full.t(), action), dim=1).to(self.device)
        input_1 = obs_full.to(self.device)
        input_2 = action.to(self.device)
        # critic_input = torch.cat((obs_full, action), dim=1).to(self.device)
        q = agent.critic(input_1, input_2)
        
         # Priorized Experience Replay
        # aux = abs(q - y.detach()) + 0.1 #we introduce a fixed small constant number to avoid priorities = 0.
        # aux = np.matrix(aux.detach().numpy())
        # new_priorities = np.sqrt(np.diag(aux*aux.T))
        
        # import pdb; pdb.set_trace()
        # Compute critic loss
        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        # Compute critic loss
        loss_mse = torch.nn.MSELoss()
        critic_loss = loss_mse(q, y.detach())
        
        # Minimize the loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        #update actor network using policy gradient
        # Compute actor loss
        agent.actor_optimizer.zero_grad()
        # make input to agent
        curr_q_input = self.maddpg_agent[agent_number].actor(obs[agent_number].to(self.device),0)
        # use Gumbel-Softmax sample
        # curr_q_input = gumbel_softmax(curr_q_input, hard = True) # this should be used only if the action is discrete (for example in comunications, but in general the action is not discrete)
        # detach the other agents to save computation
        # saves some time for computing derivative
        # q_input = [ self.maddpg_agent[i].actor(ob.to(self.device)) if i == agent_number \
        #            else self.maddpg_agent[i].actor(ob.to(self.device)).detach()
        #            for i, ob in enumerate(obs) ]
        q_input = [ curr_q_input if i == agent_number \
                   else self.maddpg_agent[i].actor(ob.to(self.device),0).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        # q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        # q_input2 = torch.cat((obs_full.to(self.device), q_input), dim=1)
        q_input2 = obs_full.to(self.device)
        actor_loss = -agent.critic(q_input2,q_input).mean() # get the policy gradient
        actor_loss += (curr_q_input).mean()*1e-3 #modification from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py
        
        # Minimize the loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)
        # return (new_priorities)

    def update_targets(self):
        """soft update targets"""
        # ----------------------- update target networks ----------------------- #
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




