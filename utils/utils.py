import torch
import os
import random
import numpy as np
import json
import time
import time
from functools import wraps


def preprocess_function(examples, tokenizer, max_length=128):
    return {
        **tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_length
        ),
        "labels": examples["label"],
    }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

    
    
def save_args(args, output_dir, time, name):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_{time}.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs) 
        end_time = time.time() 
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result

    return wrapper


def process_val_samples(val_samples):
    """Process validation samples to remove answers (content after ####)."""
    processed_samples = []
    for sample in val_samples:
        processed_sample = sample.copy()
        # Remove content after '####' in the answer
        if 'answer' in processed_sample and '####' in processed_sample['answer']:
            processed_sample['answer'] = processed_sample['answer'].split('####')[0].strip()
        processed_samples.append(processed_sample)
    return processed_samples