import matplotlib.pyplot as plt
import re

def read_losses_from_log(file_path):
    """
    Reads loss values from a log file.

    Args:
    file_path (str): The path to the log file.

    Returns:
    list: A list of loss values.
    """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'current best loss: ([0-9.]+)', line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
    return losses

def plot_losses(log_files):
    """
    Plots the loss values from multiple log files.

    Args:
    log_files (list): A list of log file paths.
    """
    plt.figure(figsize=(10, 6))

    for file in log_files:
        losses = read_losses_from_log(file)
        plt.plot(losses, label=file)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.savefig('loss_comparison.png')

# Usage example:
# log_files = ['gcg_system.log', 'gcg_no_system.log', 'gcg_no_system_eos.log', 'gcg_system_eos.log']
log_files = ['gcg_system.log', 'gcg_system_eos.log']  
plot_losses(log_files)
