''' An example of evluating the trained models in RLCard
'''
import os
from itertools import combinations

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
        print(model_path, position)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def evaluate(env, models, seed=0, num_games=2000):
    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(seed)

    # Make the environment with seed
    env = rlcard.make(env, config={'seed': seed})

    # Load models
    agents = []
    for position, model_path in enumerate(models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, num_games)
    for position, reward in enumerate(rewards):
        print(position, models[position], reward)

if __name__ == '__main__':
    
    leduc_holdem = {
        'DMC': 'src/experiments/dmc_result/leduc-holdem/0_32000.pth',
        'DQN': 'src/experiments/leduc-holdem_DQN_result/model.pth',
        'NFSP': 'src/experiments/leduc-holdem_NFSP_result/model.pth',
    }
    
    limit_holdem = {
        'DMC': 'src/experiments/dmc_result/limit-holdem/0_28800.pth',
        'DQN': 'src/experiments/limit-holdem_DQN_result/model.pth',
        'NFSP': 'src/experiments/limit-holdem_NFSP_result/model.pth',
    }
    
    no_limit_holdem = {
        'DMC': 'src/experiments/dmc_result/no-limit-holdem/0_28800.pth',
        'DQN': 'src/experiments/no-limit-holdem_DQN_result/model.pth',
        'NFSP': 'src/experiments/no-limit-holdem_NFSP_result/model.pth',
    }
    
    games = [leduc_holdem, limit_holdem, no_limit_holdem]
    envs = ['leduc-holdem', 'limit-holdem', 'no-limit-holdem']
    for i, game in enumerate(games):
        models = game.values()
        pairs = {comb for comb in combinations(models, r=2)}
        for pair in pairs:
            print(envs[i])
            # print(pair)   
            evaluate(envs[i],pair)

    