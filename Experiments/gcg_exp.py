import sys
import os

# Add the path to the BOOST folder to sys.path
sys.path.append(os.path.abspath('../BOOST'))
from BOOST.Attack_GCG.run_gcg import gcg_attack
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=0, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='allenai/tulu-2-dpo-7b',
                        help='target model path')
    parser.add_argument("--control_string_length", type=int, default=20)
    parser.add_argument("--max_attack_steps", type=int, default=500)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_attack_attempts", type=int, default=1)
    parser.add_argument("--max_prompts_in_single_attack", type=int, default=1)
    parser.add_argument("--max_successful_prompt", type=int, default=1)
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10)
    parser.add_argument("--run_index", type=int, default=0)

    args = parser.parse_args()
    gcg_attack(args)
