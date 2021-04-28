from env_wrapper import SubprocVecEnv, DummyVecEnv
import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

def make_parallel_env(n_rollout_threads, scenario, seed=1, num_agents=3, benchmark = False):
    #print('Make parallel env')
    def get_env_fn(rank):
        #print('Get env fn')
        def init_env():
            #print('Init env')
            # env = make_env("simple_adversary")
            env = make_env(scenario, num_agents=num_agents, benchmark = benchmark)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
#    if n_rollout_threads == 1:
#        return DummyVecEnv([get_env_fn(0)])
#    else:
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_env(scenario_name, num_agents=3, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    
    # load scenario from script
    #print('load scenario')
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    #print('Create world')
    world = scenario.make_world(num_agents)
    # create multiagent environment
    #print('Create multiagent environment')
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback = scenario.done)

    return env