import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMWrapper():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 ):
        # wrap the model to control the hidden states during computation
        self.model = model