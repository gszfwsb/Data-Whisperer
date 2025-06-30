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

# Set matplotlib to use high quality rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

from utils.utils import save_json
from metrics.metric import METRICS
from prompt import DATASET_PROMPTS, format_qwenvl_message_to_qa
from pruner import Pruner


class DataWhisperer_Qwen2_5VL_Pruner(Pruner):
    def __init__(self, args: Any) -> None:
        self.args = args
        self.accelerator = Accelerator()
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_path, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(self.args.model_path)
        self.tokenizer = self.processor.tokenizer

        # self.model = self.accelerator.prepare(self.model)
        self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
        if hasattr(self.model, "module"):
            self.model = self.model.module
        self.model.eval()

        self.dataset = self.args.dataset

    def generate_demonstrations(self, train_set, selected_indices, prompt_template):
        demonstrations = ""
        demo_list = []
        for idx in selected_indices:
            example = train_set[idx]
            qa_pair =  prompt_template(example)[0] # "we only want one round of conversation"
            demonstration = qa_pair[0] + "\n" + qa_pair[1]
            image_path = qa_pair[2]
            demo_list.append((demonstration, image_path))
            demonstrations += demonstration

        return demonstrations, demo_list

    def extract_predictions(self, responses_section):
        """
        Extracts answers in the format:
        Question i: <answer text>
        If no such formatted answers are found, returns the entire response.
        """
        predictions = []
        pattern_qa = (
            r"Question\s+(\d+):\s*"   # Question number and colon
            r"(.*?)"                  # Non-greedy match for answer
            r"(?=\n\s*\n|$)"          # Until the next blank line or end of text
        )

        matches_qa = re.findall(pattern_qa, responses_section, re.DOTALL | re.IGNORECASE)

        if matches_qa:
            predictions.extend(answer.strip() for _, answer in matches_qa)
        else:
            predictions.append(responses_section.strip())

        return predictions

    def visualize_causal_mask(self, causal_mask, save_path=None, max_size=512):
        """
        Visualize the causal attention mask
        
        Args:
            causal_mask: torch.Tensor of shape [batch, heads, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
            save_path: Optional path to save the visualization
            max_size: Maximum size for visualization (to avoid memory issues with large sequences)
        """
        # Convert to numpy and handle the tensor shape
        if isinstance(causal_mask, torch.Tensor):
            mask = causal_mask.detach().cpu().numpy()
        else:
            mask = causal_mask
            
        # Take the first batch and first head if multiple dimensions
        if mask.ndim == 4:
            mask = mask[0, 0]  # [seq_len, seq_len]
        elif mask.ndim == 3:
            mask = mask[0]     # [seq_len, seq_len]
            
        seq_len = mask.shape[0]
        
        # Subsample if the sequence is too long for visualization
        if seq_len > max_size:
            step = seq_len // max_size
            mask = mask[::step, ::step]
            seq_len = mask.shape[0]
            
        # Convert large negative values to 0 (masked) and 0 to 1 (visible)
        # The mask uses -inf (or very large negative values) for masked positions
        binary_mask = np.where(mask < -1e10, 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_mask, cmap='Blues', origin='upper')
        plt.title(f'Causal Attention Mask Visualization\n(Sequence Length: {seq_len})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
        
        # Add grid for better readability
        if seq_len <= 100:
            plt.grid(True, alpha=0.3)
            
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Causal mask visualization saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()

    def visualize_attention_maps_with_boundaries(
        self, 
        prompt_attentions, 
        batch_idx, 
        prompt_components, 
        images, 
        n_i, n_d, n_r, 
        demo_len, 
        total_prompt_len,
        encoding
    ):
        """
        Visualize attention maps for all layers with proper boundaries and image token positions.
        
        Args:
            prompt_attentions: Tuple of attention tensors for each layer
            batch_idx: Index of current batch item
            prompt_components: Tuple of (instruction, demonstration, response) texts
            images: List of images for this batch item
            n_i: Number of instruction tokens
            n_d: Number of demonstration tokens  
            n_r: Number of response tokens
            demo_len: List of lengths for each demonstration
            total_prompt_len: Total length of the prompt
            encoding: Tokenizer encoding output
        """
        if not hasattr(self.args, 'save_attention_visualizations') or not self.args.save_attention_visualizations:
            return
            
        inst, demo, response = prompt_components
        IMAGE_TOKEN = "<|image_pad|>"
        
        # Create output directory for attention visualizations
        vis_dir = os.path.join(self.args.output_filtered_path, "attention_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Count image tokens in each section
        inst_imgs_num = inst.count(IMAGE_TOKEN)
        demo_imgs_num = demo.count(IMAGE_TOKEN)
        response_imgs_num = response.count(IMAGE_TOKEN)
        
        # Calculate section boundaries
        boundaries = {
            'instruction': (0, n_i),
            'demonstration': (n_i, n_i + n_d),
            'response': (n_i + n_d, n_i + n_d + n_r)
        }
        
        # Calculate individual demo boundaries within demonstration section
        demo_boundaries = []
        demo_start = n_i
        for demo_length in demo_len:
            demo_boundaries.append((demo_start, demo_start + demo_length))
            demo_start += demo_length
        
        # Find image token positions
        image_positions = self._find_image_token_positions(encoding, batch_idx, IMAGE_TOKEN)
        
        # Visualize attention for each layer
        num_layers = len(prompt_attentions)
        for layer_idx in range(num_layers):
            layer_attention = prompt_attentions[layer_idx][batch_idx]  # Shape: (num_heads, seq_len, seq_len)
            
            # Average over attention heads
            avg_attention = torch.mean(layer_attention, dim=0)  # Shape: (seq_len, seq_len)
            
            # Slice to actual prompt length
            attention_matrix = avg_attention[:total_prompt_len, :total_prompt_len].detach().cpu().numpy()
            
            # Create visualization
            self._create_attention_visualization(
                attention_matrix,
                layer_idx,
                boundaries,
                demo_boundaries,
                image_positions,
                vis_dir,
                batch_idx,
                total_prompt_len
            )
            
        print(f"Attention visualizations saved to: {vis_dir}")
    
    def _find_image_token_positions(self, encoding, batch_idx, image_token):
        """Find positions of image tokens in the tokenized sequence."""
        # Get the tokenized input_ids for this batch item
        input_ids = encoding.input_ids[batch_idx]
        
        # Get the image token ID
        image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        
        # Find all positions where image tokens appear
        image_positions = []
        for pos, token_id in enumerate(input_ids):
            if token_id == image_token_id:
                image_positions.append(pos.item() if torch.is_tensor(pos) else pos)
        
        return image_positions
    
    def _add_section_braces(self, ax, boundaries, seq_len):
        """Add braces to indicate section boundaries on X and Y axes."""
        colors = {'instruction': 'red', 'demonstration': 'blue', 'response': 'green'}
        
        # Get axis limits
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        
        # Add braces on X-axis (bottom)
        brace_offset_x = seq_len * 0.08  # Offset from the main plot
        for section, (start, end) in boundaries.items():
            if start < seq_len:
                end = min(end, seq_len)
                mid_pos = (start + end) / 2
                
                # Draw brace
                self._draw_brace(ax, start, end, ylim[0] + brace_offset_x, 'horizontal', colors[section])
                
                # Add label
                ax.text(mid_pos, ylim[0] + brace_offset_x * 1.5, section.capitalize(), 
                       ha='center', va='bottom', color=colors[section], 
                       fontsize=12, fontweight='bold')
        
        # Add braces on Y-axis (left)
        brace_offset_y = seq_len * 0.08  # Offset from the main plot
        for section, (start, end) in boundaries.items():
            if start < seq_len:
                end = min(end, seq_len)
                mid_pos = (start + end) / 2
                
                # Draw brace
                self._draw_brace(ax, start, end, xlim[0] - brace_offset_y, 'vertical', colors[section])
                
                # Add label
                ax.text(xlim[0] - brace_offset_y * 1.5, mid_pos, section.capitalize(), 
                       ha='right', va='center', color=colors[section], 
                       fontsize=12, fontweight='bold', rotation=90)
    
    def _draw_brace(self, ax, start, end, offset, orientation, color):
        """Draw a brace to indicate a section."""
        if orientation == 'horizontal':
            # Draw horizontal brace (for X-axis)
            # Main horizontal line
            ax.plot([start, end], [offset, offset], color=color, linewidth=2)
            # Start vertical line
            ax.plot([start, start], [offset - 1, offset + 1], color=color, linewidth=2)
            # End vertical line  
            ax.plot([end, end], [offset - 1, offset + 1], color=color, linewidth=2)
        else:  # vertical
            # Draw vertical brace (for Y-axis)
            # Main vertical line
            ax.plot([offset, offset], [start, end], color=color, linewidth=2)
            # Start horizontal line
            ax.plot([offset - 1, offset + 1], [start, start], color=color, linewidth=2)
            # End horizontal line
            ax.plot([offset - 1, offset + 1], [end, end], color=color, linewidth=2)
    
    def _create_attention_visualization(
        self, 
        attention_matrix, 
        layer_idx, 
        boundaries, 
        demo_boundaries, 
        image_positions, 
        save_dir, 
        batch_idx, 
        seq_len
    ):
        """
        Create a comprehensive attention visualization with boundaries and annotations.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))  # Make it much larger
        
        # Main attention heatmap
        im1 = ax1.imshow(attention_matrix, cmap='viridis', aspect='auto', origin='upper')
        ax1.set_title(f'Layer {layer_idx} Attention Matrix\n(Batch {batch_idx}, Seq Length: {seq_len})', fontsize=16)
        ax1.set_xlabel('Key Position', fontsize=14)
        ax1.set_ylabel('Query Position', fontsize=14)
        
        # Add colorbar for attention magnitude
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Attention Score', rotation=270, labelpad=15, fontsize=12)
        
        # Add attention values on the heatmap (only for smaller matrices to avoid clutter)
        if seq_len <= 50:  # Only show values for smaller matrices
            for i in range(min(seq_len, attention_matrix.shape[0])):
                for j in range(min(seq_len, attention_matrix.shape[1])):
                    value = attention_matrix[i, j]
                    # Format to 2 significant digits
                    if value >= 0.01:
                        text = f'{value:.2f}'
                    elif value >= 0.001:
                        text = f'{value:.3f}'
                    else:
                        text = f'{value:.1e}'
                    ax1.text(j, i, text, ha='center', va='center', 
                            color='red', fontsize=8, fontweight='bold')
        
        # Add section braces on axes
        self._add_section_braces(ax1, boundaries, seq_len)
        
                 # Don't mark image token positions to preserve color visibility
        
        # Second subplot: Focus on response-to-demonstration attention
        demo_start, demo_end = boundaries['demonstration']
        response_start, response_end = boundaries['response']
        
        if response_start < seq_len and demo_start < seq_len:
            response_to_demo = attention_matrix[response_start:min(response_end, seq_len), 
                                             demo_start:min(demo_end, seq_len)]
            
            im2 = ax2.imshow(response_to_demo, cmap='plasma', aspect='auto', origin='upper')
            ax2.set_title(f'Layer {layer_idx}: Response→Demonstration Attention\n(Response tokens attend to Demo tokens)', fontsize=16)
            ax2.set_xlabel('Demonstration Token Position (relative)', fontsize=14)
            ax2.set_ylabel('Response Token Position (relative)', fontsize=14)
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Attention Score', rotation=270, labelpad=15, fontsize=12)
            
            # Add attention values on the focused heatmap (if small enough)
            if response_to_demo.shape[0] <= 30 and response_to_demo.shape[1] <= 30:
                for i in range(response_to_demo.shape[0]):
                    for j in range(response_to_demo.shape[1]):
                        value = response_to_demo[i, j]
                        # Format to 2 significant digits
                        if value >= 0.01:
                            text = f'{value:.2f}'
                        elif value >= 0.001:
                            text = f'{value:.3f}'
                        else:
                            text = f'{value:.1e}'
                        ax2.text(j, i, text, ha='center', va='center', 
                                color='red', fontsize=8, fontweight='bold')
            
            # Draw individual demo boundaries in the focused view
            demo_offset = 0
            for i, demo_length in enumerate([end - start for start, end in demo_boundaries]):
                if demo_offset + demo_length <= response_to_demo.shape[1]:
                    ax2.axvline(x=demo_offset, color='white', linestyle='--', linewidth=1, alpha=0.8)
                    ax2.axvline(x=demo_offset + demo_length, color='white', linestyle='--', linewidth=1, alpha=0.8)
                    
                    # Add demo labels
                    if demo_length > 5:  # Only label if demo is long enough
                        ax2.text(demo_offset + demo_length/2, -2, f'Demo {i+1}', 
                                ha='center', va='top', color='white', fontsize=12, fontweight='bold')
                demo_offset += demo_length
        
        plt.tight_layout()
        
                 # Save the figure as vector format (PDF) and high-res PNG
         save_path_pdf = os.path.join(save_dir, f'attention_layer_{layer_idx}_batch_{batch_idx}.pdf')
         save_path_png = os.path.join(save_dir, f'attention_layer_{layer_idx}_batch_{batch_idx}.png')
         
         plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
         plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create a summary statistics plot
        self._create_attention_statistics_plot(attention_matrix, layer_idx, boundaries, demo_boundaries, save_dir, batch_idx)
    
    def _create_attention_statistics_plot(self, attention_matrix, layer_idx, boundaries, demo_boundaries, save_dir, batch_idx):
        """Create statistical analysis plots for attention patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Attention distribution across sections
        section_attention = {}
        for section, (start, end) in boundaries.items():
            end = min(end, attention_matrix.shape[0])
            if start < attention_matrix.shape[0]:
                section_scores = attention_matrix[start:end, :].mean(axis=0)
                section_attention[section] = section_scores.mean()
        
        ax1.bar(section_attention.keys(), section_attention.values(), color=['red', 'blue', 'green'], alpha=0.7)
        ax1.set_title(f'Layer {layer_idx}: Average Attention by Section')
        ax1.set_ylabel('Average Attention Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Response attention to individual demonstrations
        response_start, response_end = boundaries['response']
        demo_start, demo_end = boundaries['demonstration']
        
        if response_start < attention_matrix.shape[0] and demo_start < attention_matrix.shape[1]:
            demo_attention_scores = []
            demo_labels = []
            
            for i, (d_start, d_end) in enumerate(demo_boundaries):
                if d_end <= attention_matrix.shape[1]:
                    response_slice = attention_matrix[response_start:min(response_end, attention_matrix.shape[0]), d_start:d_end]
                    if response_slice.size > 0:
                        demo_attention_scores.append(response_slice.mean())
                        demo_labels.append(f'Demo {i+1}')
            
            if demo_attention_scores:
                ax2.bar(demo_labels, demo_attention_scores, color='cyan', alpha=0.7)
                ax2.set_title(f'Layer {layer_idx}: Response Attention to Each Demo')
                ax2.set_ylabel('Average Attention Score')
                ax2.tick_params(axis='x', rotation=45)
        
        # 3. Attention entropy across sequence positions
        attention_entropy = []
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i, :]
            # Add small epsilon to avoid log(0)
            row = row + 1e-10
            entropy = -np.sum(row * np.log(row))
            attention_entropy.append(entropy)
        
        ax3.plot(attention_entropy, color='purple', linewidth=2)
        ax3.set_title(f'Layer {layer_idx}: Attention Entropy by Position')
        ax3.set_xlabel('Sequence Position')
        ax3.set_ylabel('Attention Entropy')
        ax3.grid(True, alpha=0.3)
        
        # Add section boundaries to entropy plot
        colors = {'instruction': 'red', 'demonstration': 'blue', 'response': 'green'}
        for section, (start, end) in boundaries.items():
            ax3.axvline(x=start, color=colors[section], linestyle='--', alpha=0.7, label=f'{section.capitalize()}')
        ax3.legend()
        
        # 4. Attention magnitude distribution
        ax4.hist(attention_matrix.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title(f'Layer {layer_idx}: Attention Score Distribution')
        ax4.set_xlabel('Attention Score')
        ax4.set_ylabel('Frequency')
        ax4.axvline(x=attention_matrix.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {attention_matrix.mean():.4f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the statistics plot as vector format and high-res PNG
        save_path_svg = os.path.join(save_dir, f'attention_stats_layer_{layer_idx}_batch_{batch_idx}.svg')
        save_path_png = os.path.join(save_dir, f'attention_stats_layer_{layer_idx}_batch_{batch_idx}.png')
        
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def predict_batch(
        self,
        batch_val_samples: List[List[Dict[str, str]]],
        batch_demo_list: List[List[str]],
        return_attention_scores: bool = False,
    ) -> List[List[str]]:
        prompts = []
        batch_images = []
        prompts_comp = []

        for demonstration_pairs, val_samples in zip(batch_demo_list, batch_val_samples):
            prompt, images, prompt_comp = self._prepare_model_inputs(demonstration_pairs, val_samples)
            if prompt is None:
                prompts.append("")
                batch_images.append([])
                prompts_comp.append(("", "", ""))
                continue
            
            prompts.append(prompt)
            batch_images.append(images)
            prompts_comp.append(prompt_comp)

        # Generate in batch
        with torch.no_grad():
            # Tokenize the prompts in batch
            encoding = self.processor(
                text=prompts,
                images=batch_images,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                max_length=self.args.max_token,
            ).to(self.accelerator.device)

            prompt_length = encoding.input_ids.size(1)
            max_new_tokens = self.args.max_token - prompt_length
            
            if max_new_tokens <= 0:
                self.accelerator.print(f"{max_new_tokens}:max_new_tokens<0", flush=True)
                # Return empty predictions and attention scores if applicable
                empty_preds = [[""] * len(val_samples) for val_samples in batch_val_samples]
                if return_attention_scores:
                    return empty_preds, [[] for _ in batch_val_samples]
                return empty_preds

            # Single generate call to get both sequences and attentions
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id, # 151645
                output_attentions=return_attention_scores,
                return_dict_in_generate=True,
            )
            
        # Decode batch outputs
        # The generated sequences are the part of the output after the prompt
        generated_sequences = outputs.sequences[:, prompt_length:]
        generated_texts = self.processor.batch_decode(generated_sequences, skip_special_tokens=True)

        # Extract predictions for each batch
        batch_predictions = []
        for generated_text in generated_texts:
            # The generated text is already the response, no need to split by "assistant"
            responses_section = generated_text.strip()
            predictions = self.extract_predictions(responses_section)
            batch_predictions.append(predictions)

        if return_attention_scores:
            if self.args.attn_layer is not None:
                layer = self.args.attn_layer
            else:
                layer = -1 # Default to the last layer

            prompt_attentions = outputs.attentions[0] # corresponding to the first generated token

            # prompt_attentions is a tuple with length of num_layers
            # each element is a tensor with shape (batch_size, num_heads, seq_len, seq_len)
            
            # Select the specified layer and sum over the heads
            attn_score = torch.sum(prompt_attentions[layer], dim=1).to(dtype=torch.float16) # (batch_size, seq_len, seq_len)

            attn_layer = []
            IMAGE_TOKEN = "<|image_pad|>"
            for idx in range(len(prompts_comp)):  # batch_size_parallel
                inst, demo, response = prompts_comp[idx]
                images = batch_images[idx]
                inst_imgs_num = inst.count(IMAGE_TOKEN)
                demo_imgs_num = demo.count(IMAGE_TOKEN)
                response_imgs_num = response.count(IMAGE_TOKEN)

                if not inst and not demo and not response: # Skip failed samples
                    attn_layer.append([])
                    continue
                
                demo_list = batch_demo_list[idx]

                n_i_text = self.processor(
                    text=inst,
                    images=images[:inst_imgs_num],
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    max_length=self.args.max_token,
                ).to(self.accelerator.device)
                n_d_text = self.processor(
                    text=demo,
                    images=images[inst_imgs_num:inst_imgs_num+demo_imgs_num],
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    max_length=self.args.max_token,
                ).to(self.accelerator.device)
                n_r_text = self.processor(
                    text=response,
                    images=images[inst_imgs_num+demo_imgs_num:inst_imgs_num+demo_imgs_num+response_imgs_num],
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    max_length=self.args.max_token,
                ).to(self.accelerator.device)
                n_i = n_i_text.input_ids.size(1)
                n_d = n_d_text.input_ids.size(1)
                n_r = n_r_text.input_ids.size(1)

                # Recalculate demo_len for each demonstration, including image tokens
                demo_len = []
                image_ptr = inst_imgs_num
                for _demo_text, _ in demo_list:
                    image_cnt = _demo_text.count(IMAGE_TOKEN)
                    _demo_len = self.processor(
                        text=_demo_text,
                        images=images[image_ptr:image_ptr+image_cnt],
                        return_tensors="pt",
                        truncation=False,
                        padding="longest",
                        max_length=self.args.max_token,
                    ).to(self.accelerator.device)
                    image_ptr += image_cnt
                    demo_len.append(_demo_len.input_ids.size(1))

                # The total length used for slicing should be based on actual tokenized length
                # from the attention mask to be robust against right padding.
                total_prompt_len = encoding.attention_mask[idx].sum().item()
                
                start_of_demo_tokens = n_i
                end_of_demo_tokens = start_of_demo_tokens + n_d

                # Slice the attention matrix for the current example from the batch
                attn = attn_score[idx, :total_prompt_len, :total_prompt_len]

                response_start_token = end_of_demo_tokens
                response_end_token = response_start_token + n_r
                demo_to_response = attn[
                    response_start_token:response_end_token, start_of_demo_tokens:end_of_demo_tokens
                ]  

                demo_attn = []
                demo_idx = 0
                for i in range(len(demo_list)):
                    single_demo_to_response = demo_to_response[
                        :, demo_idx : demo_idx + demo_len[i]
                    ]

                    # Normalize by the area of the attention slice
                    norm_factor = (demo_len[i] * n_r) # divide by the rectangle area on the attention map (the Fig. 2 in the paper)
                    if norm_factor > 0:
                        demo_attn.append(single_demo_to_response.sum() / norm_factor)
                    else:
                        demo_attn.append(torch.tensor(0.0, device=attn.device))

                    demo_idx += demo_len[i]

                # Visualize attention maps for all layers with boundaries

                self.visualize_attention_maps_with_boundaries(
                    prompt_attentions, 
                    idx, 
                    prompts_comp[idx], 
                    batch_images[idx], 
                    n_i, n_d, n_r, 
                    demo_len, 
                    total_prompt_len,
                    encoding
                )
                
                attn_layer.append(demo_attn)

            return batch_predictions, attn_layer

        return batch_predictions

    def _prepare_model_inputs(self, demonstration_pairs, val_samples):
        prompt_template, instruction, val_inst, task_inst = DATASET_PROMPTS[f'{self.args.model_type}_{self.args.dataset}']

        # Prepare demonstrations
        demonstrations = []
        image_paths = []
        for demo, img_path in demonstration_pairs:
            demonstrations.append(demo)
            image_paths.append(img_path)

        # Prepare validation questions
        val_texts = []
        val_img_paths = []
        for i, sample in enumerate(val_samples):
            question, _, image = format_qwenvl_message_to_qa(sample)[0]
            val_texts.append(f'Question {i + 1}: {question.replace("Question: ","")}')
            val_img_paths.append(image)

        # Construct prompt
        inst, demo, response = (
            instruction,
            "\n".join(demonstrations),
            val_inst + "\n".join(val_texts) + task_inst,
        )
        prompt = inst + demo + response

        # Collect images
        all_image_paths = image_paths + val_img_paths

        try:
            IMAGE_BASE_DIR = "/obs/users/benhao/llava-en-zh-2k"
            images = [Image.open(os.path.join(IMAGE_BASE_DIR, p)).convert("RGB") for p in all_image_paths]
        except FileNotFoundError as e:
            self.accelerator.print(f"Error loading image: {e}", flush=True)
            return None, None, None

        return prompt, images, (inst, demo, response)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        val_set: Optional[List[Dict[str, Any]]] = None,
        use_kfold: bool = False,
    ) -> str:
        total_size = len(dataset)
        score = torch.zeros(
            total_size, dtype=torch.float16, device=self.accelerator.device
        )
        count = torch.zeros(total_size, dtype=torch.int32, device=score.device)

        if use_kfold:
            assert (
                val_set is None
            ), "Validation set should not be provided for k-fold evaluation"
            kf = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=42)
            folds = list(kf.split(dataset))
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds, desc="K-Folds")):
                train_set = [dataset[i] for i in train_idx]
                val_set = [dataset[i] for i in val_idx]
                local_score = torch.zeros(
                    len(train_set), dtype=torch.float16, device=score.device
                )
                local_count = torch.zeros(
                    len(train_set), dtype=torch.int32, device=score.device
                )
                self._evaluate_single_fold(train_set, val_set, local_score, local_count)
                if not isinstance(train_idx, torch.Tensor):
                    train_idx = torch.tensor(
                        train_idx, dtype=torch.int32, device=score.device
                    )
                score.index_add_(0, train_idx, local_score)
                count.index_add_(0, train_idx, local_count)
                print(
                    f"Fold {fold_idx + 1}/{self.args.k_folds} evaluation completed",
                    flush=True,
                )
        else:
            assert (
                val_set is not None
            ), "Validation set should be provided for single dataset evaluation"
            local_score = torch.zeros(
                len(dataset), dtype=torch.float16, device=score.device
            )
            local_count = torch.zeros(
                len(dataset), dtype=torch.int32, device=score.device
            )
            self._evaluate_single_fold(dataset, val_set, local_score, local_count)
            score.add_(local_score)
            count.add_(local_count)

        final_score = torch.where(
            count > 0, score / count, torch.zeros_like(score, dtype=torch.float16)
        )
        sorted_idx = torch.argsort(final_score, descending=True)
        sorted_dataset_with_scores = [
            {
                **dataset[i],
                "score": final_score[i].item(),
            }
            for i in sorted_idx.tolist()
        ]

        output_path = os.path.join(self.args.output_filtered_path, f"data_whisperer_qwen_vl.json")

        save_json(output_path, sorted_dataset_with_scores)
        print(f"Fold evaluation completed. Results saved to {output_path}")
        return output_path

    def _evaluate_single_fold(
        self,
        train_set: List[Dict[str, Any]],
        val_set: List[Dict[str, Any]],
        score: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        train_size = len(train_set)
        val_size = len(val_set)

        train_set, val_set = self.accelerator.prepare(train_set, val_set)
        prompt_template, _, _, _ = DATASET_PROMPTS[f'{self.args.model_type}_{self.args.dataset}']

        # Generate training and validation batch indices
        train_batches = [
            (i, min(i + self.args.batch_train, train_size))
            for i in range(0, train_size, self.args.batch_train)
        ]
        val_batches = [
            (i, min(i + self.args.batch_test, val_size))
            for i in range(0, val_size, self.args.batch_test)
        ]

        train_pointer = 0
        val_pointer = 0
        fail = 0

        metric_function = METRICS[self.args.metric]
        
        progress_bar = tqdm(total=len(train_batches), desc="Evaluating Fold")
        while train_pointer < len(train_batches):
            batch_val_samples = []
            batch_selected_indices = []
            batch_demo_list = []
            
            batch_start_train_pointer = train_pointer
            # Prepare batch demonstrations and validation samples in parallel
            for _ in range(self.args.parallel_batches):
                if train_pointer >= len(train_batches):
                    break

                # Get train batch indices
                start_train_idx, end_train_idx = train_batches[train_pointer]
                selected_indices = list(range(start_train_idx, end_train_idx))
                batch_selected_indices.append(selected_indices)

                # Get validation batch indices
                start_test_idx, end_test_idx = val_batches[val_pointer]
                test_batch = val_set[start_test_idx:end_test_idx]

                # Generate demonstrations
                _, demo_list = self.generate_demonstrations(
                    train_set, selected_indices, prompt_template
                )
                batch_demo_list.append(demo_list)
                batch_val_samples.append(test_batch)
                # Update pointers
                train_pointer += 1
                val_pointer = (val_pointer + 1) % len(val_batches)

             # Generate predictions for the current batch
            batch_predictions, batch_attention_scores = self.predict_batch(
                batch_val_samples,
                batch_demo_list,
                return_attention_scores=True,
            )
            progress_bar.update(train_pointer - batch_start_train_pointer)

            # Update scores and counts efficiently on the GPU
            for predictions, val_samples, selected_indices, attention_scores in zip(
                batch_predictions,
                batch_val_samples,
                batch_selected_indices,
                batch_attention_scores,
            ):
                # We just pick first message as the reference
                def get_reference(val_sample):
                    for msg in val_sample['messages']:
                        if msg.get('role') == 'assistant':
                            return msg.get('content')
                    return None
                
                references = [get_reference(sample) for sample in val_samples]
                
                if not attention_scores: # Handle case where attention scores could not be computed
                    fail += 1
                    continue

                if not isinstance(attention_scores, torch.Tensor):
                    attention_scores = torch.tensor(
                        attention_scores, dtype=torch.float16, device=score.device
                    )

                weight = attention_scores / attention_scores.sum()
                
                if len(predictions) != len(references):
                    if len(predictions) > len(references):
                        predictions = predictions[: len(references)]
                    else:
                        fail += 1
                        continue

                for pred, ref in zip(predictions, references):
                    pred_score = metric_function(pred, ref)
                    if not isinstance(selected_indices, torch.Tensor):
                        # print('indices is not tensor')
                        selected_indices = torch.tensor(
                            selected_indices, dtype=torch.int64, device=score.device
                        )
                    if not isinstance(pred_score, torch.Tensor):
                        # print('scores is not tensor')
                        pred_score = torch.tensor(
                            [pred_score], dtype=torch.float16, device=score.device
                        ).expand(len(selected_indices))

                    weighted_scores = pred_score * weight

                    score.scatter_add_(0, selected_indices, weighted_scores)

                count[selected_indices] += len(references)
        
        progress_bar.close()
        print(f"Failed batches: {fail}")
        for val_sample in val_set:
            prediction = self.predict_batch(train_set, val_sample)
            
            # Correctly extract reference from conversation history
            reference = None
            if val_sample.get('messages') and isinstance(val_sample['messages'], list):
                for msg in reversed(val_sample['messages']):
                    if msg.get('role') == 'assistant':
                        reference = msg.get('content')
                        break
            
            if reference is None:
                print(f"Warning: Could not find reference answer for validation sample.")
                continue

            pred_score = metric_function(prediction, reference)

            # Assign uniform scores to all training samples for this validation run
            score += pred_score
            count += 1
        
        print(f"Evaluation for this fold completed.")

def test_causal_mask_visualization():
    """
    Test function to verify the causal mask visualization functionality.
    """
    print("Testing causal mask visualization...")
    
    # Create a sample causal mask (similar to what you showed)
    seq_len = 100
    causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)
    
    # Fill the upper triangle with -inf (masked positions)
    mask_value = -3.3895e+38
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            causal_mask[0, 0, i, j] = mask_value
    
    # Create a dummy pruner instance just for visualization
    args = Namespace(
        model_path="/obs/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct",
        dataset="test",
        model_type="qwen2_5_vl"
    )
    
    # We'll create a minimal version without loading the full model
    class DummyPruner:
        def visualize_causal_mask(self, causal_mask, save_path=None, max_size=512):
            # Same visualization code as in the main class
            if isinstance(causal_mask, torch.Tensor):
                mask = causal_mask.detach().cpu().numpy()
            else:
                mask = causal_mask
                
            if mask.ndim == 4:
                mask = mask[0, 0]  # [seq_len, seq_len]
            elif mask.ndim == 3:
                mask = mask[0]     # [seq_len, seq_len]
                
            seq_len = mask.shape[0]
            
            if seq_len > max_size:
                step = seq_len // max_size
                mask = mask[::step, ::step]
                seq_len = mask.shape[0]
                
            binary_mask = np.where(mask < -1e10, 0, 1)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(binary_mask, cmap='Blues', origin='upper')
            plt.title(f'Causal Attention Mask Visualization\n(Sequence Length: {seq_len})')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
            
            if seq_len <= 100:
                plt.grid(True, alpha=0.3)
                
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Causal mask visualization saved to: {save_path}")
            else:
                plt.show()
                
            plt.close()
    
    dummy_pruner = DummyPruner()
    save_path = "./temp_test_output/causal_mask_test.png"
    os.makedirs("./temp_test_output", exist_ok=True)
    
    dummy_pruner.visualize_causal_mask(causal_mask, save_path=save_path)
    print("Causal mask visualization test completed!")

def test_pruner():
    """
    Test function to verify the pruner's functionality on a small scale.
    """
    print("Starting pruner test...")
    args = Namespace(
        model_path="/obs/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct",
        data_path="/obs/users/benhao/llava-en-zh-2k/train_llava.json",
        val_path="/obs/users/benhao/llava-en-zh-2k/val_llava.json",
        output_filtered_path="./temp_test_output",
        metric='exact_match', 
        dataset="qwen2_5_vl_llava_1k_en", # For consistency, though not directly used in prompt creation now
        model_type="qwen2_5_vl",
        k_folds=2,
        attn_layer=11,
        max_token=2048,
        batch_train=1,
        batch_test=1,
        parallel_batches=1
    )
    
    # Create output directory
    os.makedirs(args.output_filtered_path, exist_ok=True)

    # Load a small subset of data for testing
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            full_train_data = json.load(f)
        test_train_set = full_train_data[:2]

        with open(args.val_path, 'r', encoding='utf-8') as f:
            full_val_data = json.load(f)
        test_val_set = full_val_data[:1]
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        print("Please ensure the paths in the test script are correct.")
        return

    print(f"Loaded {len(test_train_set)} training samples and {len(test_val_set)} validation samples for the test.")

    # Initialize and run the pruner
    pruner = DataWhisperer_Qwen2_5VL_Pruner(args)
    pruner.evaluate(dataset=test_train_set, val_set=None, use_kfold=True)

    print("Test finished successfully.")
    print(f"Pruned data scores saved in: {os.path.join(args.output_filtered_path, 'data_whisperer_qwen_vl.json')}")


if __name__ == "__main__":
    # Test the causal mask visualization first
    test_causal_mask_visualization()
    
    # Then run the full pruner test if needed
    # test_pruner() 