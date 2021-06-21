import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from target_pf import Target



class Scenario(BaseScenario):
    
    def make_world(self, num_agents=3, num_landmarks=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks*2)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = False
            else:
                landmark.name = 'landmark_estimation %d' % (i-num_landmarks)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.002
                
        # make initial conditions
        world.cov = np.ones(num_landmarks)
        world.error = np.ones(num_landmarks)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.55, 0.0, 0.0])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.agents[0].state.p_pos
                landmark.state.p_vel = np.zeros(world.dim_p)
        #Initailize the landmark estimated positions
        world.landmarks_estimated = [Target() for i in range(world.num_landmarks)]
        

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        if world.landmark.collide:
            for l in world.landmarks:
                if self.is_collision(l, agent):
                    rew -= 1
                    collisions +=1
            
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.
        
        for i,l in enumerate(world.landmarks_estimated):
            # reward as a function of covariance matrix (PF)
            world.cov[i] = np.sqrt((l.pf.covariance_vals[0])**2+(l.pf.covariance_vals[1])**2)
            rew -= world.cov[i]
            # reward as a function of the distance error between the landmark and its estimation (PF or LS)
            # world.error[i] = np.sqrt((l.pfxs[-1][0]-world.landmarks[i].state.p_pos[0])**2+(l.pfxs[-1][2]-world.landmarks[i].state.p_pos[1])**2) #Error from PF
            world.error[i] = np.sqrt((l.lsxs[-1][0]-world.landmarks[i].state.p_pos[0])**2+(l.lsxs[-1][2]-world.landmarks[i].state.p_pos[1])**2) #Error from LS
            rew -= world.error[i]
        
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks[:-world.num_landmarks]]
        if min(dists) > 2:
            rew -= 10
            
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i, entity in enumerate(world.landmarks):
            if i < world.num_landmarks: 
                #Update the landmarks_estiamted position using Particle Fileter
                #1:Compute radius between the agent and each landmark
                slant_range = np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2)
                # Add some systematic error in the measured range
                slant_range *= 0.99 # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                #2:Update the PF
                # world.landmarks_estimated[i].updatePF(dt=1., new_range=True, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.], update=True)
                #2b: Update the LS
                world.landmarks_estimated[i].updateLS(dt=1., new_range=True, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.])
                # Traditional plot
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(5,5))
                # plt.plot(world.landmarks_estimated[i].pf._x[0],world.landmarks_estimated[i].pf._x[2], 'r^', ms=20)
                # plt.plot(world.landmarks_estimated[i].pf.x.T[0],world.landmarks_estimated[i].pf.x.T[2], 'ro', ms=5, alpha=0.3)
                # plt.plot(world.landmarks[0].state.p_pos[0],world.landmarks[0].state.p_pos[1], 'ko', ms=6, alpha = 0.5)
                # plt.plot(world.agents[0].state.p_pos[0],world.agents[0].state.p_pos[1], 'ko', ms=6, alpha = 0.5)
                # plt.xlim(-1,1)
                # plt.ylim(-1,1)
                # plt.show()
                #3:Publish the new estimated position
                # world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].pfxs[-1][0],world.landmarks_estimated[i].pfxs[-1][2]] #Using PF
                world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].lsxs[-1][0],world.landmarks_estimated[i].lsxs[-1][2]] #Using LS
                
                #Append the position of the landmark to generate the observation state
                #Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position
                entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos - agent.state.p_pos)
                
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
    
    def done(self, agent, world):
        # episodes are done based on the agents minimum distance from a landmark.
        done = False
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks[:-world.num_landmarks]]
        if min(dists) > 2:
            done = True
        return done
