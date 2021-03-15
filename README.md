# MADDPG-AUV
This is a MADDPG algorithm to be used on particle environment styles. I use it to test my own scenarios for underwater target localization using autonomous vehicles. 

## Simple spread env
This algorithms has been used to solve the simple spread (Cooperative navigation) environment from OpenAI [link](https://github.com/openai/multiagent-particle-envs). N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions. However, I modified part of the reward function to be able to increase the training performance (i.e. the agents are receive +10 if are near a landmark).

<img src="https://github.com/imasmitja/MADDPG-AUV/blob/main/model/episode-49002.gif" width="250" height="250"/>

The observation space consists of 18 variables (for 3 agents and 3 landmakrs): X-Y positions of each landmark, X-Y positions other agents, and X-Y posiiton and X-Y velocities of itself, plus 2 communication of all other agents. Each agent receives its own, local observation. Two continuous cations are available, corresponding to movements of X and Y. The reward of each agent is shared in order to have a cooperative behaviour.

## Instructions
I have followed the next steps to set up my Windows computer to run the algorithms:

- conda create -n <env-name> python=3.6
- conda activate <env-name>
- conda install git
- conda install -c conda-forge ffmpeg
- pip install git+https://github.com/Kojoley/atari-py.git (optional)
- pip install gym==0.10.0
- pip install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch
- pip install tensorflow==2.1.0
- git clone https://github.com/openai/gym.git

  cd gym
 
  pip install -e .
  
Test (optional):
 - python examples/agents/random_agent.py

Next (optional as I implemented my own MADDPG):
- git clone https://github.com/openai/baselines.git

  cd baselines
  
  pip install -e .
  
Test (opitonal):
- python baselines/deepq/experiments/train_cartpole.py
- python baselines/deepq/experiments/enjoy_cartpole.py

Next:
- pip install tensorboardX
- pip install imageio
- pip install progressbar
- install pyglet==1.3.2

Train the NN network:
Run in CMD -> python main.py


Then, when the NN is trained you can visualize the polots on tensorBoard by:

Run in CMD -> tensorboard --logdir=./log/ --host=127.0.0.1

Run in web -> http://localhost:6006/


Clean:

remove all files in "model_dir" and "log" folders

Part of this has been obtained from [link](https://arztsamuel.github.io/en/blogs/2018/Gym-and-Baselines-on-Windows.html) and [link](https://knowledge.udacity.com/questions/131475), see them for further information.

