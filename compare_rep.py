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

chat = True
use_jailbreak_template = False
use_template_index = 2
if chat:
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

if chat:
    template =  "[INST] <<SYS>><</SYS>>\n\n{instruction} [/INST] "
    # template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
else:
    template = "{instruction}"

harmful_dataset = pd.read_csv('datasets/harmful.csv')['text'].tolist()
harmless_dataset = pd.read_csv('datasets/harmless.csv')['text'].tolist()
dataset = harmful_dataset + harmless_dataset


if use_jailbreak_template:
    jailbreak_templates = pd.read_csv('GPTFuzz/datasets/prompts/GPTFuzzer.csv')['text'].tolist()
    jailbreak_template = jailbreak_templates[use_template_index]
    dataset = [jailbreak_template.replace("[INSERT PROMPT HERE]", s) for s in dataset]

hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
batch_size = 8
rep_token = -1

train_rep = llm_read_rep(model, tokenizer, dataset, hidden_layers, template, batch_size, rep_token)

# Assuming `train_rep` is your dictionary with the hidden states
cosine_similarities = calculate_cosine_similarity(train_rep)
plot_heatmaps(chat,cosine_similarities)

average_activations = calculate_average_activation(train_rep)
plot_activation_comparison(chat,average_activations)