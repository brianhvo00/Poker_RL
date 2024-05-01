import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent, NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train():
    device = get_device()
    set_seed(0)
    env = rlcard.make('leduc-holdem', config={'seed': 0})
    
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64,64],
        q_mlp_layers=[64,64],
        device=device,
    )
    
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    
    with Logger('experiments/leduc_holdem_nfsp_result/') as logger:
        for episode in range(5000):
            agents[0].sample_episode_policy()
            
            
            trajectories, payoffs = env.run(is_training=True)
            
            trajectories = reorganize(trajectories, payoffs)
            
            for ts in trajectories[0]:
                agent.feed(ts)
            
            if episode % 100 == 0:
                logger.log_performance(episode, tournament(env, 10000)[0])
            
            csv_path, fig_path = logger.csv_path, logger.fig_path
            
    plot_curve(csv_path, fig_path, 'NFSP')
    save_path = os.path.join('experiments/leduc_holdem_nfsp_result/', 'model.pth')
    torch.save(agent, save_path)
    print('The model has been saved in {}'.format(save_path))

if __name__ == '__main__':
    print("running nfsp")
    train()