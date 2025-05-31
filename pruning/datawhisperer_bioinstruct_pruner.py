import os
import re
import time
from sklearn.model_selection import KFold
import torch
from utils.utils import save_json, timer_decorator
from metrics.metric import METRICS
from prompt import DATASET_PROMPTS
from pruner import Pruner
from typing import List, Dict, Any, Optional


def is_start_with_str(text: str, start: str) -> bool:
    return text.startswith(start)


class DataWhisperer_BioInstruct_Pruner(Pruner):
    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.dataset = self.args.dataset

    def generate_demonstrations(self, train_set, selected_indices, prompt_template):
        demonstrations = ""
        demo_list = []
        for idx in selected_indices:
            example = train_set[idx]
            demonstrations += prompt_template(example)
            demo_list.append(prompt_template(example))

        return demonstrations, demo_list

    def extract_predictions(self, responses_section):

        predictions = []

        # Question X Answer:
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
            
        # ### Question X:
        pattern_markdown = r"### Question \d+:\s*(.*?)(?=\n### Question \d+:|\Z)"
        matches_markdown = re.findall(pattern_markdown, responses_section, re.DOTALL)
        if matches_markdown:
            predictions.extend([match.strip() for match in matches_markdown])
            return predictions

        # Question X
        pattern_q = r"Question \d+:\s*(.*?)(?=\nQuestion \d+:|\Z)"
        matches_q = re.findall(pattern_q, responses_section, re.DOTALL)
        if matches_q:
            predictions.extend([match.strip() for match in matches_q])
            return predictions

        # Question X: Instruction: ... Input: ... Answer:
        pattern_instruction = r"Question \d+: Instruction: .*?Input:.*?Answer:\s*(.*?)(?=\nQuestion \d+:|\Z)"
        matches_instruction = re.findall(pattern_instruction, responses_section, re.DOTALL)
        if matches_instruction:
            predictions.extend([match.strip() for match in matches_instruction])
            return predictions

        # Answer X:
        pattern_ax = r"Answer \d+:\s*(.*?)(?=\nAnswer \d+:|\Z)"
        matches_ax = re.findall(pattern_ax, responses_section, re.DOTALL)
        if matches_ax:
            predictions.extend([match.strip() for match in matches_ax])
            return predictions

        # Answer:
        pattern_a = r"Answer:\s*(.*?)(?=\nAnswer:|\Z)"
        matches_a = re.findall(pattern_a, responses_section, re.DOTALL)
        if matches_a:
            predictions.extend([match.strip() for match in matches_a])
            return predictions

        return predictions  

    def get_attn_score(self, input_ids, attention_mask, layer_index):
        with torch.no_grad():

            input_embeds = self.model.model.embed_tokens(input_ids)  # [B, L, H]

            # Manually set position_ids and causal_mask
            cache_position = torch.arange(
                0, input_embeds.shape[1], device=input_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)

            if self.model.model.config.model_type == "mistral":
                causal_mask = self.model.model._update_causal_mask(
                    attention_mask,
                    input_embeds,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    output_attentions=True,
                )
            else:
                causal_mask = self.model.model._update_causal_mask(
                    attention_mask,
                    input_embeds,
                    past_key_values=None,
                    cache_position=cache_position,
                    output_attentions=True,
                )
                
            hidden_states = input_embeds
            if self.model.model.config.model_type == "mistral":
                position_embeds = self.model.model.layers[0].self_attn.rotary_emb(hidden_states, position_ids)
            else:
                position_embeds = self.model.model.rotary_emb(hidden_states, position_ids)

            attention = None

            for i, layer in enumerate(self.model.model.layers[: self.model.model.config.num_hidden_layers]):  # self.model.model.layers[: self.model.model.config.num_hidden_layers]
                if i == layer_index:
                    # Run forward pass for this layer only(including attention mechanism)
                    if self.model.model.config.model_type == "mistral":
                        layer_output = layer(
                            hidden_states=hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            use_cache=False,
                            output_attentions=True,
                        )
                    else:
                        layer_output = layer(
                            hidden_states=hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeds,
                            cache_position=cache_position,
                            use_cache=False,
                            output_attentions=True,
                        )
                        
                    attention = torch.sum(layer_output[1], dim=1).to(dtype=torch.float16)
                    break

                if self.model.model.config.model_type == "mistral":
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=False,
                        output_attentions=False,
                    )[0]                    
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeds,
                        cache_position=cache_position,
                        use_cache=False,
                        output_attentions=False,
                    )[0]

        return attention  # [B, L, L]

    def predict_batch(
        self,
        batch_demonstrations: List[str],
        batch_val_samples: List[List[Dict[str, str]]],
        batch_demo_list: List[List[str]],
        return_attention_scores: bool = False,
    ) -> List[List[str]]:
        prompts = []
        prompts_comp = []
        prompt_template, instruction, val_inst, task_inst = DATASET_PROMPTS[f'{self.args.model_type}_{self.args.dataset}']

        for demonstrations, val_samples in zip(batch_demonstrations, batch_val_samples):
            # Prepare prompts for each batch

            all_texts = []
            for i, sample in enumerate(val_samples):
                if sample["input"] and sample["input"].strip().upper() not in [
                    "",
                    "N/A",
                    "Not applicable",
                    "-",
                    "(No input needed)",
                ]:
                    all_texts.append(
                        f'Question {i + 1}: Instruction: "{sample["instruction"]}" Input: "{sample["input"]}"'
                    )
                else:
                    all_texts.append(
                        f'Question {i + 1}: Instruction: "{sample["instruction"]}"'
                    )

            inst, demo, response = (
                instruction,
                demonstrations,
                val_inst + "\n".join(all_texts) + task_inst,
            )

            prompt = inst + demo + response

            prompts.append(prompt)
            prompts_comp.append([inst, demo, response])

        # Generate in batch
        with torch.no_grad():
            # Tokenize the prompts in batch
            encoding = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                max_length=self.args.max_token,
            ).to(self.accelerator.device)
            prompt_length = encoding.input_ids.size(1)
            max_new_tokens = self.args.max_token - prompt_length
            if max_new_tokens <= 0:
                self.accelerator.print(f"{max_new_tokens}:max_new_tokens<0", flush=True)
                return [
                    [""] * len(val_samples) for val_samples in batch_val_samples
                ]  # Empty predictions for each batch
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode batch outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Extract predictions for each batch
        batch_predictions = []
        for generated_text, val_samples in zip(generated_texts, batch_val_samples):
            responses_section = generated_text.split("assistant")[-1].strip()
            predictions = self.extract_predictions(responses_section)
            batch_predictions.append(predictions)

        if return_attention_scores:
            # Get attention scores from the model
            if self.args.attn_layer is not None:
                layer = self.args.attn_layer
            else:
                if self.args.model_name == "Meta-Llama-3-8B-Instruct" or self.args.model_name == "Mistral-Nemo-Instruct-2407":
                    layer = 13
                elif self.args.model_name == "Qwen2.5-3B-Instruct" or self.args.model_name == "Qwen2.5-7B-Instruct":
                    layer = 16
                elif self.args.model_name == "Mistral-7B-Instruct-v0.2":
                    layer = 10
                else:
                    layer = 11

            attn_score = self.get_attn_score(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                layer_index=layer,            
            )
                
            attn_layer = []

            for idx in range(len(prompts_comp)):  # batch_size_parallel
                inst, demo, response = prompts_comp[idx]
                demo_list = batch_demo_list[idx]

                n_i = (
                    len(
                        self.tokenizer.encode(
                            inst,
                            truncation=False,
                            padding="longest",
                            max_length=self.args.max_token,
                        )
                    )
                    - 1
                )
                n_d = (
                    len(
                        self.tokenizer.encode(
                            demo,
                            truncation=False,
                            padding="longest",
                            max_length=self.args.max_token,
                        )
                    )
                    - 1
                )
                n_r = (
                    len(
                        self.tokenizer.encode(
                            response,
                            truncation=False,
                            padding="longest",
                            max_length=self.args.max_token,
                        )
                    )
                    - 1
                )
                demo_len = [
                    len(
                        self.tokenizer.encode(
                            _demo,
                            truncation=False,
                            padding="longest",
                            max_length=self.args.max_token,
                        )
                    )
                    - 1
                    for _demo in demo_list
                ]

                try:
                    pad_pos = (
                        torch.nonzero(1 - encoding.attention_mask[idx].squeeze())
                        .squeeze()[-1]
                        .item()
                    )
                    attn = attn_score[idx, pad_pos + 1 :, pad_pos + 1 :]
                except:
                    attn = attn_score[idx]

                demo_to_response = attn[
                    n_i + n_d : n_i + n_d + n_r, n_i : n_i + n_d
                ] 

                demo_attn = []
                demo_idx = 0
                for i in range(len(demo_list)):
                    single_demo_to_response = demo_to_response[
                        :, demo_idx : demo_idx + demo_len[i]
                    ]

                    demo_attn.append(
                        single_demo_to_response.sum() / (demo_len[i] * n_r)
                    )

                    demo_idx += demo_len[i]

                attn_layer.append(demo_attn)

            return batch_predictions, attn_layer

        return batch_predictions

    @torch.no_grad()
    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        val_set: Optional[List[Dict[str, Any]]] = None,
        use_kfold: bool = False,
    ) -> str:
        """
        Evaluate function for k-fold or a single dataset.
        """
        total_size = len(dataset)
        # Move score and count tensors directly to the accelerator device (GPU)
        score = torch.zeros(
            total_size, dtype=torch.float16, device=self.accelerator.device
        )
        count = torch.zeros(total_size, dtype=torch.int32, device=score.device)

        if use_kfold:
            assert (
                val_set is None
            ), "Validation set should not be provided for k-fold evaluation"
            # Create K-Fold splits
            kf = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=42)
            folds = list(kf.split(dataset))
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # Extract train and validation sets
                train_set = [dataset[i] for i in train_idx]
                val_set = [dataset[i] for i in val_idx]
                # Prepare local score and count for the current fold on GPU
                local_score = torch.zeros(
                    len(train_set), dtype=torch.float16, device=score.device
                )
                local_count = torch.zeros(
                    len(train_set), dtype=torch.int32, device=score.device
                )
                # Evaluate this fold
                self._evaluate_single_fold(train_set, val_set, local_score, local_count)
                # Efficiently synchronize local updates with global arrays on GPU
                if not isinstance(train_idx, torch.Tensor):
                    train_idx = torch.tensor(
                        train_idx, dtype=torch.int32, device=score.device
                    )
                score.index_add_(0, train_idx, local_score)
                count.index_add_(0, train_idx, local_count)
                # judge whether the two methods are the same
                print(
                    f"Fold {fold_idx + 1}/{self.args.k_folds} evaluation completed",
                    flush=True,
                )
        else:
            # For single dataset evaluation, ensure validation set is provided
            assert (
                val_set is not None
            ), "Validation set should be provided for single dataset evaluation"
            # Initialize local score and count on GPU for faster computation
            local_score = torch.zeros(
                len(dataset), dtype=torch.float16, device=score.device
            )
            local_count = torch.zeros(
                len(dataset), dtype=torch.int32, device=score.device
            )
            # Perform evaluation for the entire dataset
            self._evaluate_single_fold(dataset, val_set, local_score, local_count)
            # Efficiently synchronize local updates with global arrays on GPU
            score.add_(local_score)  # This in-place operation avoids Python loops
            count.add_(local_count)  # Similarly, this adds counts directly on GPU

        # Final scores (Compute mean score per example, avoid division by zero with torch where)
        final_score = torch.where(
            count > 0, score / count, torch.zeros_like(score, dtype=torch.float16)
        )
        # Sorting dataset based on scores
        sorted_idx = torch.argsort(final_score, descending=True)
        sorted_dataset_with_scores = [
            {
                **dataset[i],
                "score": final_score[i].item(),
            }  # Use `.item()` for extracting scalar value from tensors
            for i in sorted_idx.tolist()
        ]

        # Define output path based on whether K-fold is used
        output_path = os.path.join(self.args.output_filtered_path, f"data_whisperer.json")

        # Save results to JSON file
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
        """
        Evaluate a single fold, updating the local score and count on the GPU.
        """
        train_size = len(train_set)
        val_size = len(val_set)

        # Prepare model and data with accelerator
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

        while train_pointer < len(train_batches):
            batch_demonstrations = []
            batch_val_samples = []
            batch_selected_indices = []
            batch_demo_list = []

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
                demonstrations, demo_list = self.generate_demonstrations(
                    train_set, selected_indices, prompt_template
                )
                batch_demonstrations.append(demonstrations)
                batch_demo_list.append(demo_list)
                batch_val_samples.append(test_batch)
                # Update pointers
                train_pointer += 1
                val_pointer = (val_pointer + 1) % len(val_batches)

            # Generate predictions for the current batch
            batch_predictions, batch_attention_scores = self.predict_batch(
                batch_demonstrations,
                batch_val_samples,
                batch_demo_list,
                return_attention_scores=True,
            )

            # Update scores and counts efficiently on the GPU
            for predictions, val_samples, selected_indices, attention_scores in zip(
                batch_predictions,
                batch_val_samples,
                batch_selected_indices,
                batch_attention_scores,
            ):
                references = [sample["output"] for sample in val_samples]

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
        
        print(f"Failed batches: {fail}")
