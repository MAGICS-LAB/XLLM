import sys
import os

# Add the path to the BOOST folder to sys.path
sys.path.append(os.path.abspath('../BOOST'))
import argparse
from BOOST.Attack_ICA.ica import ICA_attack
from fastchat.model import add_model_args
from BOOST.utils.constants import openai_key, claude_key, gemini_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICA Attack')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--few_shot_num', type=int, default=1, help='The number of few shot examples')
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--early_stop", action="store_true", help="early stop when the attack is successful")
    parser.add_argument("--eos_num", type=int, default=20, help="max number of eos tokens")
    add_model_args(parser)

    args = parser.parse_args()
    args.openai_key = openai_key
    ICA_attack(args)