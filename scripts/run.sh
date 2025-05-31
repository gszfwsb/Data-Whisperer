BASE_PATH="."
METHOD=datawhisperer
GPUS=1

# Set datasets
DATASET=dialogsum # Support bioinstruct, gsm8k, dialogsum

# Set metrics
METRIC=rouge-L # Support rouge-L, exact_match

# Set model configurations
MODEL_TYPE=llama3_8b  # Support llama3_8b, qwen, mistral
MODEL=Llama-3-8B-Instruct  # Support Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Mistral-Nemo-Instruct-2407, Mistral-7B-Instruct-v0.2
MODEL_PATH= # <YOUR_MODEL_PATH> 

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
    > ${RESULT_DIR}/info.log 2>&1 
    
echo "Pruning script finished!"