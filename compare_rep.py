import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import pandas as pd
from read_rep import *
import csv
import argparse
import random


def get_dataset(subset, harmful_dataset, harmless_dataset, jailbreak_templates):
    if subset['harmful']:
        base_dataset = harmful_dataset
    else:
        base_dataset = harmless_dataset
        
    if 'use_jailbreak_template' in subset and subset['use_jailbreak_template']:
        if not subset['different_jb_templates']:
            if subset['random_jailbreak_templates']:
                jb_template = random.choice(jailbreak_templates)
            else:
                jb_template = jailbreak_templates[subset['use_template_index']]
            sub_dataset = [jb_template.replace("[INSERT PROMPT HERE]", s) for s in base_dataset]
        else:
            sub_dataset = []
            for i in range(len(base_dataset)):
                jb_template = random.choice(jailbreak_templates)
                sub_dataset.append(jb_template.replace("[INSERT PROMPT HERE]", base_dataset[i]))
    
    elif 'use_success_jb_template' in subset and subset['use_success_jb_template']:
        sub_dataset = []
        for i in range(len(base_dataset)):
            try:
                success_jb_template_path = 'datasets/jb_prompts/' + str(i) + '.csv'
                success_jb_template = pd.read_csv(success_jb_template_path)['prompt'].tolist()
                # random choose one
                jb_template = random.choice(success_jb_template)
                sub_dataset.append(jb_template.replace("[INSERT PROMPT HERE]", base_dataset[i]))
            except Exception as e:
                print(f"Error reading success jailbreak template for index {i}")
                print(e)
                print("Skipping this index")
    
    elif 'add_eos' in subset and subset['add_eos']:
        eos_num = subset['eos_num']
        sub_dataset = [s + '</s>'*eos_num for s in base_dataset]
        
    else:
        sub_dataset = base_dataset
        
    return sub_dataset


def main(args, data):
    if args.chat:
        model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
    else:
        model_name_or_path = 'meta-llama/Llama-2-7b-hf'

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto'
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

    if args.chat:
        template =  "[INST] <<SYS>><</SYS>>\n\n{instruction} [/INST] "
        # template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
    else:
        template = "{instruction}"

    harmful_dataset = pd.read_csv('datasets/harmful.csv')['text'].tolist()
    harmless_dataset = pd.read_csv('datasets/harmless.csv')['text'].tolist()
    jailbreak_templates = pd.read_csv('datasets/GPTFuzzer.csv')['text'].tolist()

    dataset = []
    for subset in data:
        dataset += get_dataset(subset, harmful_dataset, harmless_dataset, jailbreak_templates)
    
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    batch_size = 8
    rep_token = -1

    train_rep = llm_read_rep(model, tokenizer, dataset, hidden_layers, template, batch_size, rep_token)

    save_path = get_dir_name(data) + 'chat-' if args.chat else get_dir_name(data) + 'foundation-'
    # Assuming `train_rep` is your dictionary with the hidden states
    cosine_similarities = calculate_cosine_similarity(train_rep)
    plot_heatmaps(save_path,cosine_similarities)

    average_activations = calculate_average_activation(train_rep)
    plot_activation_comparison(save_path,average_activations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Representation Visualization')
    parser.add_argument('--chat', type=bool, default=True, help='Use chat model or the foundation model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for reading representations')
    parser.add_argument('--rep_token', type=int, default=-1, help='Token to extract representations from')
    parser.add_argument('--read_rep_method', type=str, default='vanilla', help='Method to read representations')
    args = parser.parse_args()
    
    dataset1 = {
        'harmful': True,
        'use_jailbreak_template': False,
    }
    
    dataset2 = {
        'harmful': True,
        'use_success_jb_template': True,
    }
    
    dataset3 = {
        'harmful': True,
        'different_jb_templates': True,
    }
    
    dataset4 = {
        'harmful': True,
        'add_eos': True,
        'eos_num': 10,
    }
    
    dataset5 = {
        'harmful': False,
        'add_eos': True,
        'eos_num': 10,
    }
    
    dataset6 = {
        'harmful': False,
    }
    
    data = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
    
    main(args, data)