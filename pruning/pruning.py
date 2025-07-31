from datawhisperer_bioinstruct_pruner import DataWhisperer_BioInstruct_Pruner
from datawhisperer_dialog_pruner import DataWhisperer_Dialog_Pruner
from datawhisperer_gsm_pruner import DataWhisperer_GSM_Pruner
from datawhisperer_qwen2_5_vl_pruner import DataWhisperer_Qwen2_5VL_Pruner
import argparse
import time
import torch
import threading
import os

def get_pruner(dataset, method='datawhisperer'):
    if  method == 'datawhisperer':
        pruner_map = {
            "bioinstruct": DataWhisperer_BioInstruct_Pruner,
            "dialogsum": DataWhisperer_Dialog_Pruner,
            "gsm8k": DataWhisperer_GSM_Pruner,
            "llava_1k": DataWhisperer_Qwen2_5VL_Pruner,
        }
    return pruner_map.get(dataset)

def monitor_cuda_memory(stop_event, peak_memory_list, device_index=0):
    device = torch.device(f"cuda:{device_index}")
    peak_memory = 0
    while not stop_event.is_set():
        current_memory = torch.cuda.memory_allocated(device) / 1024**2 
        peak_memory = max(peak_memory, current_memory)
        time.sleep(0.1)  
    peak_memory_list.append(peak_memory)

def run_pruning(args):
    Pruner = get_pruner(args.dataset, args.method)
    pruner = Pruner(args)
    pruner.do_pruning()

if __name__ == "__main__":

    # Benhao: uncomment this to debug the code
    # import debugpy
    # debugpy.listen(5679)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--model_type", type=str, default='llama3_8b')
    parser.add_argument("--model_name", type=str, default='Meta-Llama-3-8B-Instruct')
    parser.add_argument("--data_path", type=str, required=True, help="Path to the train dataset (JSON)")
    parser.add_argument("--val_path", type=str, default='', help="Path to the val dataset (JSON) if none using k-fold")
    parser.add_argument("--method", type=str, default='icl', help="selecting pruning method")

    parser.add_argument("--dataset", type=str, required=True, help="selecting dataset")
    parser.add_argument("--parallel_batches", type=int, default=5, help="Batch size for parallel inference")
    parser.add_argument("--batch_train", type=int, default=5, help="Batch size for training examples")
    parser.add_argument("--batch_test", type=int, default=8, help="Batch size for validation examples")
    parser.add_argument("--max_token", type=int, default=8192, help="Maximum tokens for input and output combined")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-validation")
    parser.add_argument("--metric", type=str, required=True, help="Metric name for evaluation")
    parser.add_argument("--output_filtered_path", type=str, required=True, help="Path to save filtered training data")
    parser.add_argument("--attn_layer", type=int, default=None)
    parser.add_argument("--memory_output_file", type=str, default="cuda_memory_usage.txt", help="File to save peak CUDA memory usage")
    parser.add_argument("--gpu_index", type=int, default=0, help="Index of the GPU to monitor (default: 0)")
    parser.add_argument("--log_level", type=str, default='INFO', help="Logging level") 
    parser.add_argument("--save_attention_visualizations", type=bool, default=True, help="Save attention visualizations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and PyTorch installation.")

    stop_event = threading.Event()
    peak_memory_list = []
    monitor_thread = threading.Thread(target=monitor_cuda_memory, args=(stop_event, peak_memory_list, args.gpu_index))
    monitor_thread.start()

    start_time = time.time()
    run_pruning(args)
    execution_time = time.time() - start_time

    stop_event.set()
    monitor_thread.join()

    peak_memory_mb = peak_memory_list[0] if peak_memory_list else 0
    memory_output_file = os.path.join(args.output_filtered_path, "cuda_memory_usage.txt")
    with open(memory_output_file, 'w') as f:
        f.write(f"Peak CUDA memory allocated (GPU {args.gpu_index}): {peak_memory_mb:.2f} MB\n")
        f.write(f"Execution time: {execution_time:.2f} seconds\n")

    print(f"Peak CUDA memory allocated (GPU {args.gpu_index}): {peak_memory_mb:.2f} MB saved to {args.memory_output_file}")
    print(f"Execution time: {execution_time:.2f} seconds")