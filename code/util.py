#!/usr/bin/env python3
# coding: utf-8
"""
Other utilities for some of the scripts

Author: Daniel Hernández-Lobato
"""
import json
import os
import torch
import numpy as np


def create_path(folder):
    # Check if the folder path exists
    if not os.path.exists(folder):
        # Create the folder path if it did not exist
        os.makedirs(folder)


def reset_random_state(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        
        print(f"[OK] Configuration loaded successfully.")
        return config_data

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"There was an error decoding the file {file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def rescale_arguments(args):
    """Maps a point from [0,1]^5 to MLP hyperparameters"""
    width, batch_size, alpha, learning_rate_init, max_iters = args

    width = int(np.round(np.power(2, 2 + width * 6)))
    batch_size = int(np.round(np.power(2, 2 + batch_size * 6)))
    alpha = np.power(10, -8 + alpha * 8)
    learning_rate_init = np.power(10,-5 + learning_rate_init * 5)
    max_iters = int(np.round(5 + (50 - 5) * max_iters))

    return {
        "depth": 2,
        "width": width,
        "batch_size": batch_size,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "max_iters": max_iters,
    }
