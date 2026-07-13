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
