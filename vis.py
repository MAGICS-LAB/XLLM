from operator import index
import pandas as pd
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.transformers_neuron_view import GPT2Model, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel

from bertviz.neuron_view import show


harmful = pd.read_csv('dataset/harmful.csv', index_col=False)
harmless = pd.read_csv(
    'dataset/harmful_minimal_change.csv', index_col=False)

model_type = 'gpt2'
model_version = 'meta-llama/Llama-2-7b-chat-hf'
model = AutoModel.from_pretrained(
    model_version, output_attentions=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(
    model_version, do_lower_case=True, use_fast=True)
for i in range(len(harmful)):

    sentence_a = harmful.iloc[i]['text']
    sentence_b = harmless.iloc[i]['text']

    s = show(model, model_type, tokenizer, sentence_a, sentence_b,
             display_mode='dark', html_action='return')
    with open("results/neuron_view_harm_"+str(i)+".html", 'w') as file:
        file.write(s.data)
