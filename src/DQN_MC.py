import os
import argparse

import torch

import rlcard

from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard import models
from rlcard.agents import DQNAgent

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )
    
    print("num_players", env.num_players)
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64,64],
        device=device,
    )
    # Initialize the agent and use random agents as opponents
    agents = [agent]
    for pos in range(1, env.num_players):
        agent = torch.load('/Users/brianvo/Documents/GitHub/Poker_RL_yikes/src/experiments/dmc_result/limit-holdem/1_32000.pth', map_location=device)
        agent.set_device(device)
        agents.append(agent)
    env.set_agents(agents)

    # Start training
    dir = f'{args.log_dir}/{args.env}_DQN_MC_result'
    with Logger(dir) as logger:
        for episode in range(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'DQN')

    # Save model
    save_path = os.path.join(dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    print("running dqn")
    parser = argparse.ArgumentParser("DQN_MC in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'leduc-holdem',
            'limit-holdem',
            'no-limit-holdem',
        ],
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments',
    )

    args = parser.parse_args()
    
    print(args)
    train(args)