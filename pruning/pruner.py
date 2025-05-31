import os
import re
from accelerate import Accelerator
from sklearn.model_selection import KFold
import torch
from utils.utils import load_json, save_json, timer_decorator
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import time
from utils.utils import load_json, save_json, timer_decorator
from metrics.metric import METRICS
from prompt import DATASET_PROMPTS
from abc import ABC, abstractmethod

class Pruner(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
        self.model = self.accelerator.prepare(self.model)
        # self.model = self.model.half()
        if hasattr(self.model, "module"):
            self.model = self.model.module
        self.model.eval()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.vocab_size = len(self.tokenizer)
        self.model.to(self.accelerator.device)
 
    def k_fold_split(self, dataset):
        """Split data into k folds."""
        kf = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=42)
        folds = [(train_idx, val_idx) for train_idx, val_idx in kf.split(dataset)]
        return folds
    
    @abstractmethod
    @timer_decorator
    def predict_batch(self, demonstrations, val_samples):
        """Generate predictions for a batch."""
        pass

    @torch.no_grad()    
    def generate_demonstrations(self, train_set, selected_indices, prompt_template):
        demonstrations = ""
        for idx in selected_indices:
            example = train_set[idx]
            demonstrations += prompt_template(example)
        # self.accelerator.print(f"Generated demonstrations: {demonstrations}", flush=True)
        return demonstrations

    @torch.no_grad()
    def get_prompt_template(self):
        """Provide the dataset-specific prompt template."""
        prompt_template, instruction = DATASET_PROMPTS[self.args.dataset]
        return prompt_template, instruction
    
    @abstractmethod   
    def evaluate(self, dataset, val_set, use_kfold):
         raise NotImplementedError("Subclasses must implement `predict_batch`")
    
    @timer_decorator
    def do_pruning(self):
        # Check if validation set is provided
        if self.args.val_path:
            train_set = load_json(self.args.data_path)
            val_set = load_json(self.args.val_path)
            # Call evaluate_fold in subclass
            output_path = self.evaluate(train_set, val_set, use_kfold=False)
            self.accelerator.print(f"Filtered training data saved to {output_path}", flush=True)
        else:
            dataset = load_json(self.args.data_path)
            # Call evaluate_fold in subclass with k-fold logic
            output_path = self.evaluate(dataset, use_kfold=True)
            self.accelerator.print(f"K-Fold pruning completed. Filtered training data saved to {output_path}", flush=True)
