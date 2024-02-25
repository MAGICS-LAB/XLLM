import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # for debugging
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from glob import glob
import json
from tqdm import tqdm

GEMMA_7B_PROMPT = {
    "description": "GEMMA 7B chat one shot prompt",
    "prompt": '''<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model'''
}


def create_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to('cuda')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer


def evaluate(args):
    model, tokenizer = create_model(args.model)
    
    # read the csv file
    inputs = pd.read_csv(args.path)['optim_prompts'].tolist()
    print(inputs)
    results = []
    template = GEMMA_7B_PROMPT
    for prompt in tqdm(inputs):
        prompt = template['prompt'].format(instruction=prompt)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
        num_input_tokens = inputs.input_ids.shape[1]
        decoded_output = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
        # print(decoded_output)
        results.append(decoded_output)
        print(decoded_output)


    # save the results into a csv file, if exists, delete it
    if os.path.exists(args.output):
        os.remove(args.output)
    df = pd.DataFrame(results, columns=['text'])
    df.to_csv(args.output, index=False)
    print(f"Results are saved into {args.output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GCG Results')
    parser.add_argument('--path', type=str, default='/home/jys3649/projects/LLM_MMR/Results/google/gemma-7b-it/GCG_eos/0.csv', help='The path of the csv file')
    parser.add_argument('--output', type=str, default='/home/jys3649/projects/LLM_MMR/gcg_quick_test.csv', help='The path of the output file')
    parser.add_argument('--model', type=str, default='google/gemma-7b-it', help='the model name')
    parser.add_argument('--with_sys', type=bool, default=False, help='whether to include the system response')
    args = parser.parse_args()
    
    evaluate(args)