import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def load_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            # only append the last row if the file has multiple rows
            data.append(df.iloc[-1:])
    if len(data) != 128:
        raise ValueError('The number of files is not 128')
    return pd.concat(data)  

def calculate_success_rates(data, step_intervals):
    rates = []
    total = 128
    for step in step_intervals:
        successful = (data['step'] <= step).sum() 
        rate = successful / total 
        rates.append(rate)
    return rates

def draw(args):
    fuzzer_results = f'../Results/{args.model_path}/GPTFuzzer'
    fuzzer_eos_results = f'../Results/{args.model_path}/GPTFuzzer_eos'
    fuzzer_data = load_data(fuzzer_results)
    fuzzer_eos_data = load_data(fuzzer_eos_results)
    
    step_intervals = range(0, args.steps+1, args.step_interval)
    fuzzer_rates = calculate_success_rates(fuzzer_data, step_intervals)
    fuzzer_eos_rates = calculate_success_rates(fuzzer_eos_data, step_intervals)
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(step_intervals, fuzzer_rates, label='fuzzer')
    plt.plot(step_intervals, fuzzer_eos_rates, label='fuzzer_eos')
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.title('Success Rates: Fuzzer vs Fuzzer_EOS')
    plt.legend()
    plt.grid(True) 
    save_folder = f'{args.save_path}/{args.model_path}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/fuzzer_vs_fuzzer_eos.png')
    print(f'Figure saved at {save_folder}/fuzzer_vs_fuzzer_eos.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw GPTFuzzer attack')
    parser.add_argument('--model_path', type=str, default='google/gemma-2b-it',
                        help='target model path')
    parser.add_argument('--steps', type=int, default=1000, help='The number of steps')
    parser.add_argument('--step_interval', type=int, default=10, help='The interval of steps')
    parser.add_argument('--save_path', type=str, default='.',)


    args = parser.parse_args()
    draw(args)