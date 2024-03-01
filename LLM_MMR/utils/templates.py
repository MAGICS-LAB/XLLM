LLAMA2_PROMPT_GCG = {
    "description": "Llama 2 GCG prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

'''
}

LLAMA2_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}

LLAMA2_PROMPT_no_sys = {
    "description": "Llama 2 prompt without system message",
    "prompt": '''[INST] <<SYS>>

<</SYS>>

{instruction} [/INST] '''
}

MPT_7B_PROMPT_GCG = {
    "description": "MPT 7B GCG prompt",
    "prompt": '''<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
'''
}

MPT_7B_PROMPT = {
    "description": "MPT 7B chat one shot prompt",
    "prompt": '''<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant'''
}

MPT_7B_PROMPT_no_sys = {
    "description": "MPT 7B GCG prompt",
    "prompt": '''<|im_start|>system
<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant'''
}

GEMMA_7B_PROMPT_GCG = {
    "description": "GEMMA 7B GCG prompt",
    "prompt": '''<start_of_turn>user
'''
}

GEMMA_7B_PROMPT = {
    "description": "GEMMA 7B chat one shot prompt",
    "prompt": '''<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model'''
}

# aligner_7B_PROMPT_GCG = {
#     "description": "Aligner 7B GCG prompt",
#     "prompt": '''<|im_start|>system
# - You are a helpful assistant chatbot trained by MosaicML.
# - You answer questions.
# - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
# <|im_start|>user
# '''
# }

# aligner_7B_PROMPT = {
#     "description": "Aligner 7B chat one shot prompt",
#     "prompt": '''<|im_start|>system
# - You are a helpful assistant chatbot trained by MosaicML.
# - You answer questions.
# - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
# <|im_start|>user
# {instruction}<|im_end|>
# <|im_start|>assistant'''
# }



def get_templates(model_path, func):
    if 'Llama-2' in model_path:
        if func == 'no_sys':
            return LLAMA2_PROMPT_no_sys
        elif func == 'GCG':
            return LLAMA2_PROMPT_GCG
        elif func == 'chat':
            return LLAMA2_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "no_sys", "GCG", "chat"')
    elif 'mpt' in model_path:
        if func == 'no_sys':
            return MPT_7B_PROMPT_no_sys
        elif func == 'GCG':
            return MPT_7B_PROMPT_GCG
        elif func == 'chat':
            return MPT_7B_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "no_sys", "GCG", "chat"')
    elif 'gemma' in model_path:
        if func == 'GCG':
            return GEMMA_7B_PROMPT_GCG
        elif func == 'chat' or func == 'no_sys':
            return GEMMA_7B_PROMPT
    else:
        raise ValueError(f'Unknown model {model_path}, should be one of "Llama-2", "mpt"')
    
def get_eos(model_path):
    if 'Llama-2' in model_path or 'tulu' in model_path or 'beaver' in model_path:
        return '</s>'
    elif 'mpt' in model_path or 'gpt' in model_path or 'Qwen' in model_path or 'falcon' in model_path:
        return '<|endoftext|>'
    elif 'gemma' in model_path:
        return '<eos>'
    elif 'claude' in model_path:
      return '<EOT>'
    else:
        raise ValueError(f'Unknown model {model_path}, plz set the eos token manually')
    
def get_end_tokens(model_path):
    if 'Llama-2' in model_path:
        return ' [/INST] '
    elif 'mpt' in model_path:
        return '<|im_end|>\n<|im_start|>assistant'
    elif 'gemma' in model_path:
        return '<end_of_turn>\n<start_of_turn>model'
    else:
        raise ValueError(f'Unknown model {model_path}, plz set the end token manually')
    
if __name__ == '__main__':
    print(get_templates('meta-llama/Llama-2-7b-chat-hf', 'no_sys'))