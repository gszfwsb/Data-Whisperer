# How to run DataWhisperer on multi-modal dataset with Qwen2.5-VL

We use llava-en-zh-2k dataset as an example

- Step1: Setup Environment 
`conda env create -f data_whisper.yml -n my_env_name`


- Step2: Download llava-en-zh-2k dataset from https://huggingface.co/datasets/BUAADreamer/llava-en-zh-2k

- Step3: run `cd /path/to/Data-Whisperer/data/llava_1k/ && python ./read_parquet.py` to process the data

- Step4: `cd /path/to/Data-Whisperer/ && chmod +x ./scripts/run_llava_2k.sh && ./scripts/run_llava_2k.sh`