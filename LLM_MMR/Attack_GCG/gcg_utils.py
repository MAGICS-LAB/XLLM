import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, MptForCausalLM)

def get_nonascii_toks(model_path, tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    ascii_toks = []
    if 'Llama-2' in model_path:
        # append 0 to 259
        non_ascii_toks = list(range(3, 259))
        
        for i in range(259, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                non_ascii_toks.append(i)
            else:
                ascii_toks.append(i)
    elif 'mpt' in model_path:
        for i in range(2, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                non_ascii_toks.append(i)
            else:
                ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(non_ascii_toks, device=device), torch.tensor(ascii_toks, device=device)


def verify_input(model_path):
    if 'Llama-2' in model_path:
        return [518, 29914, 25580, 29962, 29871]
    elif 'mpt' in model_path:
        return [50278, 187, 50277, 515, 5567]
    
def get_embedding_weight(model):
    """
    Creates the batch of target texts with -1 placed at the end of the sequences for padding (for masking out the loss)
    """
    # encode items and get the max length
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    elif isinstance(model, MptForCausalLM):
        return model.base_model.wte.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    elif isinstance(model, MptForCausalLM):
        return model.base_model.wte(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
def get_fixed_list(model_path):
    if 'Llama-2' in model_path:
        return ['!']
    elif 'mpt' in model_path:
        return ['this']
    else:
        raise ValueError(f'Unknown model {model_path}, should be one of "Llama-2", "mpt"')