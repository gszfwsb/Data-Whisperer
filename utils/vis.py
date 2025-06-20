import os
import re
import torch
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Optional
from PIL import Image
import json
from argparse import Namespace
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

def visualize_causal_mask(causal_mask, save_path=None, title="Causal Attention Mask"):
    """
    Visualize the causal attention mask
    
    Args:
        causal_mask: torch.Tensor of shape [batch_size, num_heads, seq_len, seq_len] or [seq_len, seq_len]
        save_path: str, optional path to save the visualization
        title: str, title for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy and handle different tensor shapes
    if isinstance(causal_mask, torch.Tensor):
        mask_np = causal_mask.detach().cpu().numpy()
    else:
        mask_np = causal_mask
    
    # Handle different dimensions
    if mask_np.ndim == 4:  # [batch_size, num_heads, seq_len, seq_len]
        mask_np = mask_np[0, 0]  # Take first batch and first head
    elif mask_np.ndim == 3:  # [num_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
        mask_np = mask_np[0]   # Take first head or first batch
    
    # Convert large negative values to 0 (masked) and small values to 1 (unmasked)
    # -3.3895e+38 is approximately negative infinity in bfloat16
    binary_mask = (mask_np > -1e10).astype(np.float32)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Raw mask values (log scale for better visibility)
    mask_for_plot = np.where(mask_np < -1e10, -40, 0)  # Convert -inf to -40 for visualization
    im1 = ax1.imshow(mask_for_plot, cmap='RdYlBu_r', aspect='auto')
    ax1.set_title(f'{title} - Raw Values (log scale)')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1, label='Mask Value')
    
    # Add grid lines every 100 tokens for better readability
    seq_len = mask_np.shape[0]
    for i in range(0, seq_len, 100):
        ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Binary mask (0 = masked, 1 = unmasked)
    im2 = ax2.imshow(binary_mask, cmap='Blues', aspect='auto')
    ax2.set_title(f'{title} - Binary Mask (Blue=Unmasked, White=Masked)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2, label='Mask Value (0=Masked, 1=Unmasked)')
    
    # Add grid lines every 100 tokens
    for i in range(0, seq_len, 100):
        ax2.axhline(y=i, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        ax2.axvline(x=i, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})
    
    # Print some statistics
    total_elements = mask_np.size
    masked_elements = np.sum(mask_np < -1e10)
    unmasked_elements = total_elements - masked_elements
    
    print(f"Causal Mask Statistics:")
    print(f"  Shape: {mask_np.shape}")
    print(f"  Total elements: {total_elements}")
    print(f"  Masked elements: {masked_elements} ({masked_elements/total_elements*100:.1f}%)")
    print(f"  Unmasked elements: {unmasked_elements} ({unmasked_elements/total_elements*100:.1f}%)")
    print(f"  Min value: {mask_np.min()}")
    print(f"  Max value: {mask_np.max()}")
    
    # Save if path provided
    if save_path:
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # plt.show()
    
    return binary_mask

def test_causal_mask_visualization():
    """
    Test the causal mask visualization function with a sample mask
    """
    # Create a sample causal mask similar to what you showed
    seq_len = 100
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    
    # Fill upper triangle with large negative values (masked positions)
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            mask[0, 0, i, j] = -3.3895e+38
    
    print("Testing causal mask visualization...")
    visualize_causal_mask(mask, save_path="./temp_test_output/causal_mask_test.png", 
                         title="Test Causal Mask")

if __name__ == "__main__":
    test_causal_mask_visualization()