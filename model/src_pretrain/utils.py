import yaml
import pandas as pd
import joblib
from datetime import datetime
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

# Function to load config files
def config_load():
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameter file not found in path')
    return config

# Function to load json files
def load_json(file_path):
    data = open(file_path)
    return pd.read_json(data)

# Function to load tsv files
def load_tsv(file_path):
    data = open(file_path)
    return pd.read_csv(data, sep="\t")

# Function to dump data into json
def dump_json(data, file_path):
    data.to_json(file_path)
    
def pickle_load(file_path):
    """Function to load pickle files"""
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """Function to dump data into pickle"""
    joblib.dump(data, file_path)
    
# Function that return the current time
def time_stamp():
    return datetime.now()

print(CONFIG_DIR)