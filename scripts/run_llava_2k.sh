BASE_PATH="."
METHOD=datawhisperer
GPUS=1

# Set datasets
DATASET=llava_1k

# Set metrics
METRIC=exact_match # Support rouge-L, exact_match

# Set model configurations
MODEL_TYPE=qwen2_5_vl
MODEL=Qwen2.5-VL-3B-Instruct
MODEL_PATH=/obs/pretrained_models/Qwen/Qwen2.5-VL-3B-Instruct

# Set numbers of samples for demonstration and query
BATCH_TRAIN=5
BATCH_TEST=3

# Set parallel size
PARALLEL=1


# Run pruning experiment
EXP_NAME=${MODEL}_${DATASET}_${METHOD}_${METRIC}_${BATCH_TRAIN}_${BATCH_TEST}_${PARALLEL}

RESULT_DIR=${BASE_PATH}/results/pruning/${MODEL_NAME}/${EXP_NAME}

TRAIN_PATH="${BASE_PATH}/data/${DATASET}/train.json"
if [ "$DATASET" = "dialogsum" ]; then
    VAL_PATH="${BASE_PATH}/data/dialogsum/validation.json"
else
    VAL_PATH=""
fi

# TODO: set correct path for llava_1k dataset
if [ "$DATASET" = "llava_1k" ]; then
    TRAIN_PATH="Data-Whisperer/data/llava_1k/train_llava.json"
    VAL_PATH="Data-Whisperer/data/llava_1k/val_llava.json"
fi

mkdir -p ${RESULT_DIR}

MAIN_HOST=$(hostname -I | awk '{print $1}')
cat <<EOF > ${RESULT_DIR}/config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0
main_process_ip: $MAIN_HOST
main_process_port: 32343
num_machines: 1
mixed_precision: bf16
num_processes: $GPUS
EOF

export PYTHONPATH=${BASE_PATH}/pruning:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

echo "Launching pruning script with BATCH_TRAIN=$BATCH_TRAIN and BATCH_TEST=$BATCH_TEST..."
accelerate launch \
    --config_file ${RESULT_DIR}/config.yaml \
    ${BASE_PATH}/pruning/pruning.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --model_name ${MODEL} \
    --parallel_batches ${PARALLEL} \
    --data_path $TRAIN_PATH \
    --batch_train ${BATCH_TRAIN} \
    --batch_test ${BATCH_TEST} \
    --dataset ${DATASET} \
    --method ${METHOD} \
    --output_filtered_path $RESULT_DIR \
    --metric ${METRIC} \
    ${VAL_PATH:+--val_path $VAL_PATH} \
    --gpu_index 0 \
    # > ${RESULT_DIR}/info.log 2>&1 
    
echo "Pruning script finished!"