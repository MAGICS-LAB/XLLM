import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from utils.dataset import get_dataset


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
    harmless_dataset = pd.read_csv('datasets/harmful_minimal_change.csv')['text'].tolist()
    jailbreak_templates = pd.read_csv('datasets/GPTFuzzer.csv')['text'].tolist()

    dataset = []
    dataset_slice = []
    for subset in data:
        subdataset =  get_dataset(subset, harmful_dataset, harmless_dataset, jailbreak_templates)
        dataset_slice.append(len(subdataset))
        dataset += subdataset
    
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    batch_size = args.batch_size
    rep_token = args.rep_token
    save_path = get_dir_name(data) + 'chat-' if args.chat else get_dir_name(data) + 'foundation-'
    
    train_rep = llm_read_rep(model, tokenizer, dataset, hidden_layers, template, batch_size, rep_token)
    avg_cos_sim = average_cosine_similarity(train_rep)
    plot_avg_cosine_similarity(save_path,avg_cos_sim)
    diff_rep = pair_diff(train_rep)
    pca_components = perform_pca_per_hidden_state(diff_rep, n_components=1)
    
    
    plot_pca(save_path,pca_components)
    
    
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Representation Visualization')
    parser.add_argument('--chat', type=bool, default=True, help='Use chat model or the foundation model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for reading representations')
    parser.add_argument('--rep_token', type=int, default=-1, help='Token to extract representations from')
    parser.add_argument('--read_rep_method', type=str, default='vanilla', help='Method to read representations')
    args = parser.parse_args()
    
    dataset1 = {
        'harmful': True,
    }
    
    dataset2 = {
        'harmful': False,
    }
   
    data = [dataset1, dataset2]
    
    main(args, data)