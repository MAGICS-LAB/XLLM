import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import wandb
from collections import OrderedDict
from utils.dataset import get_dataset
import random
import argparse
import csv
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


class SimpleAverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


def attach_act_hooks(model):

    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]

            act_dict[name] = (inp, out)
        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def kurtosis(data):
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / n
    fourth_moment = sum((x - mean) ** 4 for x in data) / n
    return fourth_moment / (var ** 2)


def main(args, data):
    if args.chat:
        model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
    else:
        model_name_or_path = 'meta-llama/Llama-2-7b-hf'

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

    if args.chat:
        template = "[INST] <<SYS>><</SYS>>\n\n{instruction} [/INST] "
        # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
    else:
        template = "{instruction}"

    harmful_dataset = pd.read_csv('datasets/harmful.csv')['text'].tolist()
    harmless_dataset = pd.read_csv('datasets/harmless.csv')['text'].tolist()
    jailbreak_templates = pd.read_csv(
        'datasets/GPTFuzzer.csv')['text'].tolist()

    dataset = []
    dataset_slice = []
    for subset in data:
        print(subset)
        subdataset = get_dataset(
            subset, harmful_dataset, harmless_dataset, jailbreak_templates)
        # dataset_slice.append(len(subdataset))
        dataset = subdataset

        act_dict = attach_act_hooks(model)
        act_inf_norms = OrderedDict()
        act_kurtoses = OrderedDict()
        batch_size = args.batch_size
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Fetching hidden states"):
                batch_inputs = dataset[i:i+batch_size]
                batch_inputs = [template.format(instruction=s)
                                for s in batch_inputs]
                model_inputs = tokenizer(
                    batch_inputs, return_tensors='pt', padding=True).to(model.device)
                outputs = model(**model_inputs, output_hidden_states=True)
                # print(outputs.keys())
                # wandb.log({"outputs": outputs.hidden_states.shape})
                num_layers = len(model.model.layers)
                ACT_KEYS = [
                    "model.norm",
                    *[f"model.layers.{j}" for j in range(num_layers)],
                    *[f"model.layers.{j}.mlp" for j in range(num_layers)],
                    *[f"model.layers.{j}.input_layernorm" for j in range(num_layers)],
                    *[f"model.layers.{j}.self_attn.v_proj" for j in range(num_layers)],
                    *[f"model.layers.{j}.post_attention_layernorm" for j in range(num_layers)],
                    *[f"model.layers.{j}.mlp.act_fn" for j in range(num_layers)],
                ]
                for name in act_dict:
                    if name in ACT_KEYS:
                        x_inp, x_out = act_dict[name]
                        x = x_out
                        if isinstance(x, list):
                            continue
                        x = x.detach().cpu().contiguous().view(x.size(0), -1)

                        # compute inf norm
                        inf_norms = x.norm(dim=1, p=np.inf)
                        if not name in act_inf_norms:
                            act_inf_norms[name] = SimpleAverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())

                        kurt = kurtosis(x)
                        if not name in act_kurtoses:
                            act_kurtoses[name] = SimpleAverageMeter()
                        for v in kurt:
                            act_kurtoses[name].update(v.item())
        metrics = OrderedDict()
        for name, v in act_inf_norms.items():
            metrics[name] = v.avg
        max_inf_norm = [v.avg for v in act_inf_norms.values()]
        max_inf_norm = max(max_inf_norm)
        max_ffn_inf_norm = max(
            v.avg for k, v in act_inf_norms.items() if ".mlp" in k)
        max_layer_inf_norm = max(
            act_inf_norms[f"model.layers.{j}"].avg for j in range(num_layers)
        )

        avg_kurtosis = sum(v.avg for v in act_kurtoses.values()
                           ) / len(act_kurtoses.values())
        max_kurtosis = max(v.avg for v in act_kurtoses.values())
        max_kurtosis_layers = max(
            act_kurtoses[f"model.layers.{j}"].avg for j in range(num_layers)
        )

        metrics["max_inf_norm"] = max_inf_norm
        metrics["max_ffn_inf_norm"] = max_ffn_inf_norm
        metrics["max_layer_inf_norm"] = max_layer_inf_norm
        metrics["avg_kurtosis"] = avg_kurtosis
        metrics["max_kurtosis"] = max_kurtosis
        metrics["max_kurtosis_layers"] = max_kurtosis_layers
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Representation Visualization')
    parser.add_argument('--chat', type=bool, default=True,
                        help='Use chat model or the foundation model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for reading representations')
    parser.add_argument('--rep_token', type=int, default=-1,
                        help='Token to extract representations from')
    parser.add_argument('--read_rep_method', type=str,
                        default='vanilla', help='Method to read representations')
    args = parser.parse_args()

    dataset1 = {
        'harmful': True,
    }

    dataset2 = {
        'harmful': False,
    }

    dataset3 = {
        'harmful': True,
        'use_jailbreak_template': True,
        'different_jb_templates': True,
    }

    dataset4 = {
        'harmful': True,
        'use_success_jb_template': True,
    }

    dataset5 = {
        'harmful': True,
        'use_gcg': True,
    }

    dataset6 = {
        'harmful': True,
        'add_eos': True,
        'eos_num': 5,
    }

    dataset7 = {
        'harmful': False,
        'add_eos': True,
        'eos_num': 5,
    }

    # data = [dataset1, dataset2, dataset3,
    #         dataset4, dataset5, dataset6, dataset7]
    data = [dataset1]

    main(args, data)
