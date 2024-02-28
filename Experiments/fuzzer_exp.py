import sys
import os

# Add the path to the LLM_MMR folder to sys.path
sys.path.append(os.path.abspath('../LLM_MMR'))
import argparse
from LLM_MMR.Attack_GPTFuzzer.gptfuzz import fuzzer_attack
from fastchat.model import add_model_args
from LLM_MMR.utils.constants import openai_key, claude_key, gemini_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=10, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=5,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="./Dataset/fuzzer_seed.csv", help="The seed path")
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10, help="The number of eos tokens")
    add_model_args(parser)

    args = parser.parse_args()
    args.openai_key = openai_key
    args.claude_key = claude_key
    args.gemini_key = gemini_key
    fuzzer_attack(args)