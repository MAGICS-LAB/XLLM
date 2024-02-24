import sys
import os

# Add the path to the LLM_MMR folder to sys.path
sys.path.append(os.path.abspath('../LLM_MMR'))
from LLM_MMR.Attack_GCG.run_gcg import gcg_attack
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=0, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='google/gemma-7b-it',
                        help='target model path')
    parser.add_argument("--control_string_length", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_attack_steps", type=int, default=4000)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--max_attack_attempts", type=int, default=20)
    parser.add_argument("--max_prompts_in_single_attack", type=int, default=1)
    parser.add_argument("--max_successful_prompt", type=int, default=1)
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10)

    args = parser.parse_args()
    gcg_attack(args)
