# Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning

This repository contains a reference implementation for ACL 2025 main paper [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](https://arxiv.org/pdf/2505.12212).

[Shaobo Wang*<sup>1,2</sup>](https://gszfwsb.github.io/), Xiangqi Jin<sup>2</sup>, Ziming Wang<sup>2,3</sup>, [Jize Wang<sup>1</sup>](https://jize-w.github.io/), Jiajun Zhang<sup>2</sup>,  
[Kaixin Li<sup>4</sup>](https://likaixin2000.github.io/), Zichen Wen<sup>2</sup>, [Zhong Li<sup>5</sup>](https://www.microsoft.com/en-us/research/people/lzhong/), [Conghui He<sup>6</sup>](https://conghui.github.io/), [Xuming Hu<sup>7</sup>](https://xuminghu.github.io/), [Linfeng Zhang†<sup>1,2</sup>](http://www.zhanglinfeng.tech/)

*Equal contribution, †Corresponding author   
<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>EPIC Lab, Shanghai Jiao Tong University <sup>3</sup>Nanyang Technological University  
<sup>4</sup>National University of Singapore <sup>5</sup>Microsoft Research Asia <sup>6</sup>Shanghai AI Laboratory  
<sup>7</sup>Hong Kong University of Science and Technology (Guangzhou)

## Pipeline
<img width="825" alt="image" src="https://github.com/user-attachments/assets/37b5958f-1c55-447a-ae54-05f30b7bc224" />

(I) **Few-shot In-Context Learning.**   
A set of demonstration and query examples is randomly sampled from the initial dataset, and an ICL prompt is constructed with a fixed instruction. The LLM to be fine-tuned generates answers for all query examples, and the average evaluation score is computed using the ground truth answers.   
(II) **Context-Aware Weighting.**   
During each iteration of few-shot ICL, we weight the scores of the demonstration examples based on their attention scores, which quantify their influence on the queries.

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
@article{wang2025datawhisperer,
  title = {Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning},
  author = {Wang, Shaobo and Jin, Xiangqi and Wang, Ziming and Wang, Jize and Zhang, Jiajun and Li, Kaixin and Wen, Zichen and Li, Zhong and He, Conghui and Hu, Xuming and Zhang, Linfeng},
  year = {2025},
  journal = {Annual Meeting of the Association for Computational Linguistics},
}
```




