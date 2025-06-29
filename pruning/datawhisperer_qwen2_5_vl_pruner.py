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
        # This method can be reused from other pruners if the output format is similar.
        # TODO: need to confirm the output format
        predictions = []
        pattern_qa = (
            r"\s*\*{0,2}"
            r"Question\s+\d+\s+Answer"
            r":?\*{0,2}"
            r"\s*"
            r"(.*?)"
            r"(?=(?:\s*\n\s*\*{0,2}Question\s+\d+\s+Answer:?\*{0,2}\s*)|$)"
        )
        matches_qa = re.findall(pattern_qa, responses_section, re.DOTALL | re.IGNORECASE)
        if matches_qa:
            predictions.extend([match.strip() for match in matches_qa])
            return predictions
        # Fallback for single response
        if not predictions:
            return [responses_section.strip()]
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
                pad_to_multiple_of=8
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
                pad_token_id=self.processor.tokenizer.eos_token_id,
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
                    # pad_to_multiple_of=8
                ).to(self.accelerator.device)
                n_d_text = self.processor(
                    text=demo,
                    images=images[inst_imgs_num:inst_imgs_num+demo_imgs_num],
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    max_length=self.args.max_token,
                    # pad_to_multiple_of=8
                ).to(self.accelerator.device)
                n_r_text = self.processor(
                    text=response,
                    images=images[inst_imgs_num+demo_imgs_num:inst_imgs_num+demo_imgs_num+response_imgs_num],
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    max_length=self.args.max_token,
                    # pad_to_multiple_of=8
                ).to(self.accelerator.device)
                n_i = n_i_text.input_ids.size(1)
                n_d = n_d_text.input_ids.size(1)
                n_r = n_r_text.input_ids.size(1)

                # Recalculate demo_len for each demonstration, including image tokens
                demo_len = []
                image_ptr = inst_imgs_num
                for _demo_text, _demo_img_path in demo_list:
                    image_cnt = _demo_text.count(IMAGE_TOKEN)
                    _demo_len = self.processor(
                        text=_demo_text,
                        images=images[image_ptr:image_ptr+image_cnt],
                        return_tensors="pt",
                        truncation=False,
                        padding="longest",
                        max_length=self.args.max_token,
                        # pad_to_multiple_of=8
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