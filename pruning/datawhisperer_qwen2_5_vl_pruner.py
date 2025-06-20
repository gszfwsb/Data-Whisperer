import os
import re
import torch
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Optional
from PIL import Image
import json
from argparse import Namespace
from tqdm.auto import tqdm

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

    def get_attn_score(self, input_ids, attention_mask, layer_index):
        with torch.no_grad():
            # For Qwen2.5-VL, the language model is under `language_model` attribute
            language_model = self.model.language_model

            # so here, if you decode the input_ids, it will be like:

            """
            ["<|im_start|>system\nA chat between a curious user and an artificial intelligence assistant. 
            The assistant gives helpful, detailed, and polite answers to the user's questions.
            <|im_end|>
            <|im_start|>user
            Below are some demonstrations of how to format your answers:
            nQuestion: What kind of bed is in the image?
            Image: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>\n\nAnswer: The image shows a set of wooden bunk beds, which is a double-story bed with one bed stacked on top of the other. 
            
            Question: Why might the dog be in the car?\nImage: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Answer: The dog might be in the car for various reasons, such as accompanying its owner on a trip or running errands, going to a veterinary clinic for a check-up, or visiting a park for recreational activities like playing or walking. It is not unusual for dog owners to bring their pets along with them in their vehicles, as this allows the dog to experience different environments and social interactions, while also giving the owner and their dog companionship during journeys. Additionally, having the dog in the car ensures that it is not left unattended at home for long periods, which can potentially lead to boredom, anxiety, or destructive behaviors. 
            
            Question: Describe the following image.\nImage: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Answer: The image features a man sitting at a table at an outdoor restaurant, ready to enjoy his meal. In front of him is a plate of pizza, and he has a knife in his hand to cut it. There are multiple chairs and dining tables surrounding him, creating a dining area with ample seating options.\n\nSeveral cups can be seen placed on the tables nearby, indicating that beverages are available to accompany the meals. In addition to the man's knife, there is a fork visible on another table. The overall scene conveys a welcoming atmosphere for diners to enjoy their food and beverages. 
            
            Question: What can be inferred about the person's eating habits or dietary preferences from this image?\nImage: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Answer: From the image showing a spoonful of rice with two pieces of broccoli, it can be inferred that the person might have a preference for a balanced diet. Rice provides carbohydrates, while broccoli is rich in vitamins, minerals, and fiber. This combination suggests that they might be trying to meet different nutritional requirements by consuming a mix of whole grains and vegetables. However, it is worth noting that this is just a snapshot and not a comprehensive view of the person's eating habits or dietary preferences. 
            
            Question: How many horses are there in the image?\nImage: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Answer: There are three horses in the image. 
            
            **Strictly use the format specified below:**
            
            Question 1 Answer: <your answer>
            #### <final answer>
            Question 2 Answer: <your answer>
            #### <final answer> 
            
            Now, based on the provided questions, respond to the following questions:
            
            Question 1: How many people are there in the image, and what are they doing?
            Image: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Question 2: What factors may influence the woman's success in flying the kite?
            Image: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
            
            Question 3: What factors could influence the woman's ability to catch the Frisbee?
            Image: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>\n<|im_end|>\n\n<|im_start|>assistant<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"]

            
            """
            
            input_embeds = language_model.embed_tokens(input_ids)

            # input_embeds.shape: torch.Size([1, 1712, 2048])
            cache_position = torch.arange(
                0, input_embeds.shape[1], device=input_embeds.device
            )
            # Manually set position_ids
            # TODO: verify the correctness of position_ids for Qwen2.5-VL
            position_ids = cache_position.view(1, -1).expand(input_ids.shape[0], -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            # Here we need to handle the position ids for images correctly

            # Prepare 4D causal mask
            causal_mask = language_model._update_causal_mask(
                attention_mask,
                input_embeds,
                cache_position,
                None,
                output_attentions=True
            )

            hidden_states = input_embeds
            position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
            attention = None

            for i, layer in enumerate(language_model.layers):
                if i == layer_index:
                    # Run forward pass for this layer only(including attention mechanism)
                    layer_output = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=True,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                        
                    attention = torch.sum(layer_output[1], dim=1).to(dtype=torch.float16)
                    break
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )[0]

        return attention  # [B, L, L]

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
                return [
                    [""] * len(val_samples) for val_samples in batch_val_samples
                ]  # Empty predictions for each batch
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        # Decode batch outputs
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        # Extract predictions for each batch
        batch_predictions = []
        for generated_text, val_samples in zip(generated_texts, batch_val_samples):
            # print(generated_text)
            responses_section = generated_text.split("assistant")[-1].strip()
            # print(f"responses_section: {responses_section}")
            predictions = self.extract_predictions(responses_section)
            batch_predictions.append(predictions)

        if return_attention_scores:
            # Get attention scores from the model
            if self.args.attn_layer is not None:
                layer = self.args.attn_layer
            else:
                layer = 11 # TODO: choose correct layer? 

            attn_layer = []
            # encoding["pixel_values"].shape = ([4200, 1176])
            attn_score = self.get_attn_score(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                layer_index=layer,
            )

            for idx in range(len(prompts_comp)):  # batch_size_parallel
                inst, demo, response = prompts_comp[idx]
                if not inst and not demo and not response: # Skip failed samples
                    attn_layer.append([])
                    continue
                
                demo_list = batch_demo_list[idx]
                
                # TODO: verify the correctness of the following code, whether they are producing correct attention scores
                # Correctly calculate token lengths including image patches
                vision_config = self.model.config.vision_config
                image_size = self.processor.image_processor.size["height"]
                patch_size = vision_config.patch_size
                num_tokens_per_image = (image_size // patch_size) ** 2

                # Number of images in demo and response parts
                num_demo_images = len([d for d in demo_list if d[1] is not None])
                num_val_images = len(batch_val_samples[idx])

                # Calculate text token lengths from components, ignoring special tokens.
                # This is an approximation, as the final prompt construction by the
                # processor might add its own template tokens.
                n_i_text = len(self.processor.tokenizer.encode(inst, add_special_tokens=False))
                n_d_text = len(self.processor.tokenizer.encode(demo, add_special_tokens=False))
                n_r_text = len(self.processor.tokenizer.encode(response, add_special_tokens=False))
                
                # Total lengths including image patches
                n_i = n_i_text
                n_d = n_d_text + num_demo_images * num_tokens_per_image
                n_r = n_r_text + num_val_images * num_tokens_per_image

                # Recalculate demo_len for each demonstration, including image tokens
                demo_len = []
                for _demo_text, _demo_img_path in demo_list:
                    text_len = len(self.processor.tokenizer.encode(_demo_text, add_special_tokens=False))
                    image_len = num_tokens_per_image if _demo_img_path is not None else 0
                    demo_len.append(text_len + image_len)

                # The total length used for slicing should be based on actual tokenized length
                # from the attention mask to be robust against padding.
                total_prompt_len = encoding.attention_mask[idx].sum().item()
                
                # Approximate start of the demo section for slicing
                # The length of tokenized instruction part is the best guess for the start of demo section
                start_of_demo_tokens = len(self.processor.tokenizer.encode(inst, add_special_tokens=False))
                end_of_demo_tokens = start_of_demo_tokens + n_d

                try:
                    # This logic correctly extracts the non-padded part of the attention matrix
                    # for left-padded sequences.
                    pad_pos = (
                        torch.nonzero(1 - encoding.attention_mask[idx].squeeze())
                        .squeeze()[-1]
                        .item()
                    )
                    attn = attn_score[idx, pad_pos + 1 :, pad_pos + 1 :]
                except:
                    # This handles the case with no padding
                    attn = attn_score[idx]

                # Slicing the attention matrix.
                # `demo_to_response` means attention from response tokens (queries) to demo tokens (keys/values)
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
                    norm_factor = (demo_len[i] * n_r)
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
        k_folds=2
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
    test_pruner() 