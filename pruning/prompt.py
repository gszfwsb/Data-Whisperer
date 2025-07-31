"""
prompt_template, instruction, val_inst, task_inst = DATASET_PROMPTS[f'{self.args.model_type}_{self.args.dataset}']
"""


import os


DATASET_PROMPTS = {

"llama3_8b_bioinstruct": (
    lambda sample: f'Instruction: {sample["instruction"]}\nInput: {sample["input"]}\nAnswer: {sample["output"]}\n',

    (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
You are a medical expert. Given an input and an instruction, your objective is to respond with the correct and concise answer based on the provided context. Ensure clarity and accuracy in your answers.\n\
### Guidelines for your response:\n\
1. **Ensure your responses are concise, clear, and focused on the provided instruction.** Provide clear, correct answers, without unnecessary elaboration.\n\
2. **Follow the logical order of questions.** Do not skip or merge responses.\n\
3. **Stay within the scope of the question.** Avoid additional commentary or unrelated details.\n\
Please respond to the questions based on the given instructions and demonstrations below.\n\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\
Below are some demonstrations of how to format your answers:\n\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Question 1 Answer: <your answer to Question 1.>\n\
Question 2 Answer: <your answer to Question 2.>\n\
(and so on...).\n\
### Now, based on the biomedical demonstrations provided, respond to the following biomedical questions:\n"""
    ),

    (
"""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    ),

),


"llama3_8b_gsm8k": (
    lambda sample: f'Question: {sample["question"]}\nAnswer:\n{sample["answer"]}\n',

    (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
You are an expert math assistant. Your role is to provide step-by-step calculations for each problem and deliver the correct final answer. Each solution should be logically structured, with no extra commentary or deviation from the required steps. \
Your responses must be concise, accurate, and in the exact format specified below. Your sole focus should be on solving the problem as efficiently as possible. Do not include any extraneous information.\n\
### Guidelines for your response:\n\
1. Your response must contain only step-by-step calculations and the final answer.\n\
2. The final output **must** be formatted as: #### <number>.\n\
Replace `<number>` with the correct final result (either an integer or a floating-point number). No deviations or alternative formats are allowed.\n\
3. Do not add any commentary, questions, greetings, or extra remarks.\n\
4. Ensure your calculations are clear, concise, and correct, but only include the steps required to arrive at the final answer.\n\
Please answer each question step by step and provide the final answer following the instructions below.\n\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Question 1 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
Question 2 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
(and so on...).\n\
### Now, based on the provided questions, respond to the following mathematical problems:\n"""
    ),

    ("""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""),
),


"llama3_8b_dialogsum": (
    lambda sample: f'Dialogue: {sample["dialogue"]}\nSummary: \n{sample["summary"]}\n',

    (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
You are an expert assistant. Your task is to provide clear, concise, and complete summaries for the given dialogues. Your summaries should accurately capture the main points of each dialogue. Avoid unnecessary details and ensure clarity.\n\
### Guidelines for your response:\n\
1. **Summarize the dialogue concisely and fully**, ensuring all main points are captured.\n\
2. **Avoid adding extra commentary or irrelevant details** that are not part of the dialogue content.\n\
3. If a dialogue is unclear, incomplete, or lacks meaningful content, respond with "No valid content to summarize."\n\
4. **Ensure every summary field is filled out.** Leaving any field blank is not allowed.\n\
Please summarize dialogues based on the given instructions and demonstrations below.\n\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Summary 1: <Your summary to Dialogue 1.>\n\
Summary 2: <Your summary to Dialogue 2.>\n\
(and so on...).\n\
### Now, based on the provided dialogues, provide concise and complete summaries for the following dialogues:\n"""
    ),

    ("""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""),

),

"qwen_bioinstruct": (
    lambda sample: f'Instruction: {sample["instruction"]}\nInput: {sample["input"]}\nAnswer: {sample["output"]}\n',

    (
"""<|im_start|>system\n\
You are a medical expert. Given an input and an instruction, your objective is to respond with the correct and concise answer based on the provided context. \n\
### Guidelines for your response:\n\
1. **Ensure your responses are concise, clear, and focused on the provided instruction.** \n\
2. **Follow the logical order of questions.** Do not skip or merge responses.\n\
3. **Avoid adding extra commentary or irrelevant details. DO NOT repeat or summarize the question.** \n\
<|im_end|>\n<|im_start|>user\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Assistant must strictly use the format specified below:**\n\
Question 1 Answer: <your answer to Question 1.>\n\
Question 2 Answer: <your answer to Question 2.>\n\
(and so on...).\n\
### Now, based on the biomedical demonstrations provided, respond to the following biomedical questions:\n"""
    ),

    (
"""<|im_end|>\n\n<|im_start|>assistant"""
    ),

),

"qwen_dialogsum": (
    lambda sample: f'Dialogue: {sample["dialogue"]}\nSummary: \n{sample["summary"]}\n',

    (
"""<|im_start|>system\n\
You are an expert assistant. Your task is to provide clear, concise, and complete summaries for the given dialogues. Your summaries should accurately capture the main points of each dialogue. Avoid unnecessary details and ensure clarity.\n\
### Guidelines for your response:\n\
1. **Summarize the dialogue concisely and fully**, ensuring all main points are captured.\n\
2. **Avoid adding extra commentary or irrelevant details** that are not part of the dialogue content.\n\
3. If a dialogue is unclear, incomplete, or lacks meaningful content, respond with "No valid content to summarize."\n\
4. **Ensure every summary field is filled out.** Leaving any field blank is not allowed.\n\
Please summarize dialogues based on the given instructions and demonstrations below.\n\
<|im_end|>\n<|im_start|>user\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Summary 1: <Your summary to Dialogue 1.>\n\
Summary 2: <Your summary to Dialogue 2.>\n\
(and so on...).\n\
### Now, based on the provided dialogues, provide concise and complete summaries for the following dialogues:\n"""
    ),

    ("""<|im_end|>\n\n<|im_start|>assistant"""),

),

"qwen_gsm8k": (
    lambda sample: f'Question: {sample["question"]}\nAnswer:\n{sample["answer"]}\n',

    (
"""<|im_start|>system\n\
You are an expert math assistant. Your role is to provide step-by-step calculations for each problem and deliver the correct final answer. Each solution should be logically structured, with no extra commentary or deviation from the required steps. \
Your responses must be concise, accurate, and in the exact format specified below. Your sole focus should be on solving the problem as efficiently as possible. Do not include any extraneous information.\n\
### Guidelines for your response:\n\
1. Your response must contain only step-by-step calculations and the final answer.\n\
2. The final output **must** be formatted as: #### <number>.\n\
Replace `<number>` with the correct final result (either an integer or a floating-point number). No deviations or alternative formats are allowed.\n\
3. Do not add any commentary, questions, greetings, or extra remarks.\n\
4. Ensure your calculations are clear, concise, and correct, but only include the steps required to arrive at the final answer.\n\
Please answer each question step by step and provide the final answer following the instructions below.\n\
<|im_end|>\n<|im_start|>user\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Question 1 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
Question 2 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
(and so on...).\n\
### Now, based on the provided questions, respond to the following mathematical problems:\n"""
    ),

    ("""<|im_end|>\n\n<|im_start|>assistant"""),
),


"mistral_bioinstruct": (
    lambda sample: f'Instruction: {sample["instruction"]}\nInput: {sample["input"]}\nAnswer: {sample["output"]}\n',

    (
"""<s>[INST]\
You are a medical expert. Given an input and an instruction, your objective is to respond with the correct and concise answer based on the provided context. Ensure clarity and accuracy in your answers.\n\
### Guidelines for your response:\n\
1. **Ensure your responses are concise, clear, and focused on the provided instruction.** Provide clear, correct answers, without unnecessary elaboration.\n\
2. **Follow the logical order of questions.** Do not skip or merge responses.\n\
3. **Stay within the scope of the question.** Avoid additional commentary or unrelated details.\n\
Please respond to the questions based on the given instructions and demonstrations below.\n\
\n\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Question 1 Answer: <your answer to Question 1.>\n\
Question 2 Answer: <your answer to Question 2.>\n\
(and so on...).\n\
### Now, based on the biomedical demonstrations provided, respond to the following biomedical questions:\n"""
    ),

    (
"""\nassistant\n[/INST]"""
    ),

),

"mistral_dialogsum": (
    lambda sample: f'Dialogue: {sample["dialogue"]}\nSummary: \n{sample["summary"]}\n',

    (
"""<s>[INST]\
You are an expert assistant. Your task is to provide clear, concise, and complete summaries for the given dialogues. Your summaries should accurately capture the main points of each dialogue. Avoid unnecessary details and ensure clarity.\n\
### Guidelines for your response:\n\
1. **Summarize the dialogue concisely and fully**, ensuring all main points are captured.\n\
2. **Avoid adding extra commentary or irrelevant details** that are not part of the dialogue content.\n\
3. If a dialogue is unclear, incomplete, or lacks meaningful content, respond with "No valid content to summarize."\n\
4. **Ensure every summary field is filled out.** Leaving any field blank is not allowed.\n\
Please summarize dialogues based on the given instructions and demonstrations below.\n\
\n\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Summary 1: <Your summary to Dialogue 1.>\n\
Summary 2: <Your summary to Dialogue 2.>\n\
(and so on...).\n\
### Now, based on the provided dialogues, provide concise and complete summaries for the following dialogues:\n"""
    ),

    ("""\nassistant\n[/INST]"""),

),

"mistral_gsm8k": (
    lambda sample: f'Question: {sample["question"]}\nAnswer:\n{sample["answer"]}\n',

    (
"""<s>[INST]\
You are an expert math assistant. Your role is to provide step-by-step calculations for each problem and deliver the correct final answer. Each solution should be logically structured, with no extra commentary or deviation from the required steps. \
Your responses must be concise, accurate, and in the exact format specified below. Your sole focus should be on solving the problem as efficiently as possible. Do not include any extraneous information.\n\
### Guidelines for your response:\n\
1. Your response must contain only step-by-step calculations and the final answer.\n\
2. The final output **must** be formatted as: #### <number>.\n\
Replace `<number>` with the correct final result (either an integer or a floating-point number). No deviations or alternative formats are allowed.\n\
3. Do not add any commentary, questions, greetings, or extra remarks.\n\
4. Ensure your calculations are clear, concise, and correct, but only include the steps required to arrive at the final answer.\n\
Please answer each question step by step and provide the final answer following the instructions below.\n\
\n\n\
Below are some demonstrations of how to format your answers:\n"""
    ),

    (
"""\n**Strictly use the format specified below:**\n\
Question 1 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
Question 2 Answer: <your step-by-step solution>\n\
#### <final answer>\n\
(and so on...).\n\
### Now, based on the provided questions, respond to the following mathematical problems:\n"""
    ),

    ("""\nassistant\n[/INST]"""),
),
} 


def convert_llava_to_qwen_multimodal_chat(sample):
    """
        {
        "messages": [
            {
                "role": "user",
                "content": "<image>What challenges may the person face in these snowy conditions?"
            },
            {
                "role": "assistant",
                "content": "The person in the image, clad in ski gear and riding skis across snow-covered ground, may potentially face challenges in terms of maintaining balance, skiing securely, and adapting to rapidly changing environmental conditions. Navigating through different terrains and snow quality might require physical strength, stamina, and adaptability. Furthermore, they may also encounter visibility problems, especially if snowfall or fog reduces visibility. Those factors can increase the risk of accidents or getting lost. In such snowy conditions, it is essential that the skier is well-equipped with appropriate gear, knowledgeable about the environment, and possibly accompanied by a buddy, a group, or an expert guide for safety purposes."
            }
        ],
        "images": [
            {
                "path": "llava_images/image_23_0.png"
            }
        ]
    },
    """
    messages = sample['messages']
    images = sample['images']
    formatted_messages = []
    img_ptr = 0
    for message in messages:
        formatted_msg = {
            "role": message['role'],
            "content": []
        }
        content = message['content']
        if "<image>" in content:
            formatted_msg['content'].append(
                {
                    "type": "image",
                    "image": images[img_ptr]['path']
                }
            )
            img_ptr += 1
            text = content.replace("<image>", "")
            formatted_msg['content'].append(
                {
                    "type": "text",
                    "text": text,
                }
            )
        else:
            formatted_msg['content'].append(
                {
                    "type": "text",
                    "text": content,
                }
            )
        formatted_messages.append(formatted_msg)
    return formatted_messages

def format_qwenvl_message_to_qa(sample):
    msgs = convert_llava_to_qwen_multimodal_chat(sample)
    qa_pairs = []
    for msg in msgs:
        if msg['role'] == 'user':
            for content in msg['content']:
                if content['type'] == 'image':
                    image_path = content['image']
                    image_text = f"Image: <|vision_start|><|image_pad|><|vision_end|>\n"
                elif content['type'] == 'text':
                    question_text = f"Question: {content['text']}\n"
            question_text =  question_text + image_text
        elif msg['role'] == 'assistant':
            for content in msg['content']:
                if content['type'] == 'text':
                    answer_text = f"Answer: {content['text']} \n"
            break # NOTE: we only want one round of conversation
    qa = (question_text, answer_text, image_path)
    qa_pairs.append(qa)
    assert len(qa_pairs) == 1, "we only want one round of conversation"
    return qa_pairs



def _format_qwen_multimodal_chat(sample, dataset_type = "llava"):
    add_vision_id = False
    add_generation_prompt = True
    
    image_count = 0
    video_count = 0
    result_parts = []
    
    if dataset_type == "llava":
        messages = convert_llava_to_qwen_multimodal_chat(sample)
    else:
        assert False, "dataset_type must be llava"
    
    if not messages or messages[0]['role'] != 'system':
        result_parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")

    for message in messages:
        result_parts.append(f"<|im_start|>{message['role']}\n")
        
        content = message['content']
        if isinstance(content, str):
            result_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'image' or 'image' in part or 'image_url' in part:
                    image_count += 1
                    if add_vision_id:
                        result_parts.append(f"Picture {image_count}: ")
                    result_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif part.get('type') == 'video' or 'video' in part:
                    video_count += 1
                    if add_vision_id:
                         result_parts.append(f"Video {video_count}: ")
                    result_parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                elif 'text' in part:
                    result_parts.append(part['text'])
        
        result_parts.append("<|im_end|>\n")

        if message['role'] == 'assistant':
            break # NOTE: we only want one round of conversation

    if add_generation_prompt:
        result_parts.append("<|im_start|>assistant\n")
        
    return "".join(result_parts)


DATASET_PROMPTS["qwen2_5_vl_llava_1k"] = (
    format_qwenvl_message_to_qa,

    # the system prompt is from: https://github.com/haotian-liu/LLaVA/issues/477
    # Qwen should actually use : You are a helpful assistant.
    (
"""<|im_start|>system\n\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<|im_end|>\n<|im_start|>user\n\
Below are some demonstrations of how to format your answers:\n"""
    ),
    (
"""**Strictly use the format specified below to answer all the questions:**\n\

Question i: <your answer>

### Now, based on the provided questions, respond to the following questions one by one:\n"""
    ),

    ("""<|im_end|>\n\n<|im_start|>assistant\n"""),
)

if __name__ == "__main__":
    # test the prompt
    import json
    llava_en_zh_2k = "/obs/users/benhao/llava-en-zh-2k"
    json_file = os.path.join(llava_en_zh_2k, "llava-en-zh-2k.json")
    with open(json_file, "r") as f:
        data = json.load(f)
    prompt_template, instruction, val_inst, task_inst = DATASET_PROMPTS["qwen2_5_vl_llava_1k_en"]
    print(prompt_template(data[0], dataset_type="llava"))
    print(instruction)
    print(val_inst)
    print(task_inst)
    exit()