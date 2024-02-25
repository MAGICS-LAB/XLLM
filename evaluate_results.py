import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from glob import glob
import json
from tqdm import tqdm

#list1 = [14]
list1 = [144]

LLAMA2_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>><</SYS>>

{instruction} [/INST] '''
}

LLAMA2_PROMPT_SYS = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}


def create_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = model.to('cuda')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer


def evaluate(args):
    model, tokenizer = create_model(args.model)
    
    # read the csv file
    inputs = pd.read_csv(args.path)['text'].tolist()
    responses = pd.read_csv(args.output)['text']
    
    if args.with_sys:
        template = LLAMA2_PROMPT_SYS
    else:
        template = LLAMA2_PROMPT
    
    for indx in tqdm(range(len(inputs))):
        if indx in list1: 
            prompt = inputs[indx]
            prompt = template['prompt'].format(instruction=prompt)
            input = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**input, max_length=256)
            num_input_tokens = input.input_ids.shape[1]
            decoded_output = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
            responses[indx] = decoded_output
        


    # save the results into a csv file, if exists, delete it
    # if os.path.exists(args.output):
    #     os.remove(args.output)
    #df = pd.DataFrame(results, columns=['text'])
    responses.to_csv(args.output, index=False)
    print(f"Results are saved into {args.output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GCG Results')
    parser.add_argument('--path', type=str, default='dataset/AdvBench_minimal_change.csv', help='The path of the csv file')
    parser.add_argument('--output', type=str, default='dataset/harmful_response.csv', help='The path of the output file')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='the model name')
    parser.add_argument('--with_sys', type=bool, default=False, help='whether to include the system response')
    args = parser.parse_args()
    
    evaluate(args)
