# Data-Whisperer
Code for ACL 2025 Main paper ["Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning"](https://arxiv.org/pdf/2505.12212).

## 🔧 Getting Started
### 🛠️ Setup
```sh
git clone https://github.com/gszfwsb/Data-Whisperer.git
cd Data-Whisperer
pip install -r requirements.txt
```
### 🧪 Experiments
To run a data selection experiment using **Data Whisperer**, please refer to `scripts/run.sh` to modify the parameters according to your requirements.

```bash
# Set dataset
DATASET=gsm8k # Support bioinstruct, gsm8k, dialogsum

# Set metric
METRIC=exact_match # Support rouge-L, exact_match

# Set model configurations
MODEL_TYPE=llama3_8b  # Support llama3_8b, qwen, mistral
MODEL=Llama-3-8B-Instruct  # Support Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Mistral-Nemo-Instruct-2407, Mistral-7B-Instruct-v0.2
MODEL_PATH= # <YOUR_MODEL_PATH> 

# Set numbers of samples for demonstration and query
BATCH_TRAIN=5
BATCH_TEST=3

# Set parallel size
PARALLEL=5
```
- DATASET: Dataset name. Support `bioinstruct`, `gsm8k`, `dialogsum`.
- METRIC: Evaluation metric for dataset. Support `rouge-L` for `bioinstruct` and `dialogsum`, and `exact_match` for `gsm8k`.
- MODEL_TYPE: Type of the model. Support `llama3_8b`, `qwen`, `mistral`.
- MODEL: Name of the model. Support `Llama-3-8B-Instruct`, `Qwen2.5-7B-Instruct`, `Mistral-Nemo-Instruct-2407`, and `Qwen2.5-3B-Instruct`, `Mistral-7B-Instruct-v0.2` for weak-to-strong experiments.
- MODEL_PATH: Path to your model.
- BATCH_TRAIN: Number of samples for demonstration in In-Context Learning. 
- BATCH_TEST: Number of samples for query in In-Context Learning. 
- PARALLEL: Parallel size.

After modifying parameters, run:
```bash 
bash scripts/run.sh 
```

Upon completion of the experiment, the scored dataset will be generated and stored in the `results/pruning` directory. You can then select data points based on the corresponding metric to construct a coreset.

## 📝 Citation
If you find **Data Whisperer** useful for your research and applications, please kindly cite using this BibTeX:
```latex
@article{wang2025data,
  title={Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning},
  author={Wang, Shaobo and Wang, Ziming and Jin, Xiangqi and Wang, Jize and Zhang, Jiajun and Li, Kaixin and Wen, Zichen and Li, Zhong and He, Conghui and Hu, Xuming and others},
  journal={arXiv preprint arXiv:2505.12212},
  year={2025}
}
```




