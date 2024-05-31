import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import pandas as pd
import argparse
from LLM_MMR.Attack_GCG.gcg import GCG
import csv
from LLM_MMR.utils.templates import get_eos

def gcg_attack(args):

    questions = pd.read_csv('Dataset/harmful.csv')['text'].tolist()[args.index]
    
    if args.add_eos:
        eos_token = get_eos(args.model_path)
        questions = questions + eos_token * args.eos_num
    args.question = questions
    print("The question is: ", questions)
    
    gcg = GCG(args)
    target = pd.read_csv('./Dataset/harmful_targets.csv')['target'].tolist()[args.index]
    optim_prompts, steps = gcg.run(target)
    
    # save the optim prompts into a csv file
    save_path = f'./Results/{args.model_path}/GCG-{args.run_index}/{args.index}.csv'
    if args.add_eos:
        save_path = f'./Results/{args.model_path}/GCG_eos-{args.run_index}/{args.index}.csv'
        
    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        #write the column name
        writer.writerow(['optim_prompts', 'steps'])
        for prompt, step in zip(optim_prompts, steps):
            writer.writerow([prompt, step])
    
    print("The optim prompts are saved.")