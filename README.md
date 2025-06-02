# The official implementation for paper Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs’ Ethical Boundaries

## Running Environment
Running the code requires a GPU with at least 40GB memory as it requires LLM inference and gradient computation (for GCG).

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
Each running script is in Scripts folders.
For example,
```bash
bash ./Scripts/run_GCG.sh
```
The running results will be saved in the Logs folder.

## Close-source LLM Probing
We provide the probing screenshots in the Probing folder.