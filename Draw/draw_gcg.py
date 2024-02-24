import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def load_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            data.append(df)
    return pd.concat(data)  

def calculate_success_rates(data, step_intervals):
    rates = []
    total = 128
    for step in step_intervals:
        successful = (data['steps'] <= step).sum() 
        rate = successful / total 
        rates.append(rate)
    return rates

def draw(args):
    gcg_results = f'../Results/{args.model_path}/GCG'
    gcg_eos_results = f'../Results/{args.model_path}/GCG_eos'
    gcg_data = load_data(gcg_results)
    gcg_eos_data = load_data(gcg_eos_results)
    
    step_intervals = range(0, args.steps+1, args.step_interval)
    gcg_rates = calculate_success_rates(gcg_data, step_intervals)
    gcg_eos_rates = calculate_success_rates(gcg_eos_data, step_intervals)
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(step_intervals, gcg_rates, label='gcg')
    plt.plot(step_intervals, gcg_eos_rates, label='gcg_eos')
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.title('Success Rates: GCG vs GCG_EOS')
    plt.legend()
    plt.grid(True) 
    save_folder = f'{args.save_path}/{args.model_path}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/gcg_vs_gcg_eos.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw GCG attack')
    parser.add_argument('--model_path', type=str, default='mosaicml/mpt-7b-chat',
                        help='target model path')
    parser.add_argument('--steps', type=int, default=1000, help='The number of steps')
    parser.add_argument('--step_interval', type=int, default=100, help='The interval of steps')
    parser.add_argument('--save_path', type=str, default='.',)


    args = parser.parse_args()
    draw(args)