import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import os

def get_hidden_states(model, model_inputs):
    with torch.no_grad():
        outputs =  model(**model_inputs, output_hidden_states=True)
    hidden_state = outputs['hidden_states']
    return hidden_state

def llm_read_rep(model, tokenizer, dataset, hidden_layers, template, batch_size, rep_token):
    hidden_states = {}
    # create the key for the hidden layer
    for layer in hidden_layers:
        hidden_states[layer] = []
    # fetch the data by batch
    for i in tqdm(range(0, len(dataset), batch_size), desc="Fetching hidden states"):
        batch_inputs = dataset[i:i+batch_size]
        batch_inputs = [template.format(instruction=s) for s in batch_inputs]
        model_inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True).to(model.device)
        batch_hidden_states = get_hidden_states(model, model_inputs)
        # store the hidden states
        for layer in hidden_layers:
            hidden_states[layer].extend(batch_hidden_states[layer][:,rep_token,:].cpu().numpy())
    print("Done fetching hidden states")
    # hidden_states = average_hidden_states(hidden_layers, hidden_states)
    return hidden_states

def calculate_cosine_similarity(hidden_states):
    """
    Calculate cosine similarity for each pair of input representations in each layer.
    """
    cosine_similarities = {}
    for layer in hidden_states.keys():
        layer_representations = np.array(hidden_states[layer])
        try:
            # Check for NaN values
            if np.isnan(layer_representations).any():
                print(f"NaN values found in layer {layer} at indices: {np.argwhere(np.isnan(layer_representations))}")
            cosine_similarities[layer] = cosine_similarity(layer_representations)
        except Exception as e:
            print(f"Error calculating cosine similarity for layer {layer}")
    return cosine_similarities

def get_dir_name(data):
    """
    create the directory name for the figures
    """
    date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f"figures/{date_string}/"
    os.makedirs(save_dir, exist_ok=True)
    data_string = json.dumps(data, indent=4)
    with open(f'{save_dir}/data.json', 'w') as f:
        f.write(data_string)
    return save_dir

def plot_heatmaps(save_path,cosine_similarities):
    """
    Plot heatmaps of cosine similarities for each layer.
    """
    for layer, similarity_matrix in cosine_similarities.items():
        plt.figure()
        sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm')
        plt.title(f"Cosine Similarity Heatmap for Layer {layer}")
        plt.xlabel("Input Index")
        plt.ylabel("Input Index")
        plt.savefig(save_path + f"cosine_similarity_layer_{layer}.png")
        plt.close()

def calculate_average_activation(hidden_states,dataset_slice=None):
    """
    Calculate the average absolute activation for each layer.
    """
    if dataset_slice == None:
        average_activations = {}
        for layer in hidden_states.keys():
            hidden_states_array = np.array(hidden_states[layer])
            average_activations[layer] = np.mean(np.abs(hidden_states_array))
        return average_activations
    else:
        average_activations = []
        start_index = 0
        end_index = 0
        for i in range(len(dataset_slice)):
            end_index += dataset_slice[i]
            sub_dataset_average_activations = {}
            for layer in hidden_states.keys():
                hidden_states_array = np.array(hidden_states[layer][start_index:end_index])
                sub_dataset_average_activations[layer] = np.mean(np.abs(hidden_states_array))
            average_activations.append(sub_dataset_average_activations)
            start_index = end_index
        return average_activations

def plot_activation_comparison(save_path,average_activations):
    """
    Plot the average absolute activations for each layer.
    """
    layers = list(average_activations.keys())
    avg_activations = [average_activations[layer] for layer in layers]
    plt.figure()
    plt.bar(layers, avg_activations)
    plt.title("Average Absolute Activation per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Average Activation")
    plt.savefig(save_path + "layer_activation_comparison.png")
    plt.close()
    
def plot_activation_line(save_path,average_activations):
    """
    Plot the average absolute activations for each layer.
    The y-axis is the average absolute activation.
    The x-axis is the layer index
    Put all the average activations in one plot.
    """
    layers = list(average_activations[0].keys())
    for i in range(len(average_activations)):
        avg_activations = [average_activations[i][layer] for layer in layers]
        plt.plot(layers, avg_activations, label=f"dataset {i}")
    plt.title("Average Absolute Activation per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Average Activation")
    plt.legend()
    plt.savefig(save_path + "layer_activation_line.png")
    plt.close()
