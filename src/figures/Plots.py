import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    leduc = ['src/experiments/leduc-holdem_DQN_result/performance.csv',
            'src/experiments/leduc-holdem_NFSP_result/performance.csv',
            'src/experiments/leduc_holdem_cfr_result/leduc-holdem_CFR_result/performance.csv']
    
    limit = ['src/experiments/limit-holdem_DQN_result/performance.csv',
            'src/experiments/limit-holdem_NFSP_result/performance.csv']
    
    no_limit = ['src/experiments/no-limit-holdem_DQN_result/performance.csv',
                'src/experiments/no-limit-holdem_NFSP_result/performance.csv']
    
    DMC = ['src/experiments/dmc_result/leduc-holdem/logs.csv',
           'src/experiments/dmc_result/limit-holdem/logs.csv',
           'src/experiments/dmc_result/no-limit-holdem/logs.csv']

    games = [leduc, limit, no_limit]
    game_name = ['Leduc Holdem', 'Limit Holdem', 'No Limit Holdem']
    
    colors = ['red', 'green', 'blue']
    labels = ['DQN', 'NFSP', 'CFR']

    for i, game in enumerate(games):
        plt.figure()  # Create a new figure for each game
        for j, path in enumerate(game):
            df = pd.read_csv(path)
            plt.plot(df['episode'], df['reward'], label=labels[j], color=colors[j])
            
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward (Big Blind/Hand)')
        plt.legend()
        plt.savefig(f'src/figures/{game_name[i]}plot.png')
        
    DMC_game_name = ['Leduc Holdem DMC', 'Limit Holdem DMC', 'No Limit Holdem DMC']

    for i, path in enumerate(DMC):
        df = pd.read_csv(path)
        df = df.iloc[:1500]
        plt.figure()  # Create a new figure for each game
        plt.plot(df['# _tick'], df['mean_episode_return_1'], label='DMC', color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward (Big Blind/Hand)')
        plt.title(DMC_game_name[i])
        plt.legend()
        plt.savefig(f'src/figures/{DMC_game_name[i]}plot.png')