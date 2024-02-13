import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import pandas as pd
import argparse
from gcg import GCG
import csv

def main(args):

    questions = pd.read_csv('/home/jys3649/projects/xllm/datasets/harmful.csv')['text'].tolist()[args.index]
    if args.add_eos:
        questions = questions + '</s>'*8
    args.question = questions
    gcg = GCG(args)
    target = pd.read_csv('/home/jys3649/projects/xllm/datasets/harmful_target.csv')['target'].tolist()[args.index]
    optim_prompts = gcg.run(target)
    # save the optim prompts into a csv file
    save_path = f'/home/jys3649/projects/xllm/datasets/gcg/{args.index}.csv'
    if args.add_eos:
        save_path = f'/home/jys3649/projects/xllm/datasets/gcg_eos/{args.index}.csv'
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        #write the column name
        writer.writerow(['optim_prompts'])
        for prompt in optim_prompts:
            writer.writerow([prompt])
    
    print("The optim prompts are saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=0, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='mutate model path')
    parser.add_argument("--control_string_length", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--max_attack_attempts", type=int, default=10)
    parser.add_argument("--max_prompts_in_single_attack", type=int, default=10)
    parser.add_argument("--max_successful_prompt", type=int, default=20)
    parser.add_argument("--add_eos", type=bool, default=True)

    args = parser.parse_args()
    main(args)
