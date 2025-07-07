
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PER_DEVICES_BS=4
GRAD_ACC=16
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
ACCELERATE_CONFIG_PATH=accelerate_config/gpu${GPU_COUNT}_zero3_acc${GRAD_ACC}.yaml

REPORT_TO=wandb
PORT=33445

export WANDB__SERVICE_WAIT=600
export WANDB_MODE=online


DATA_PATH=
EVAL_DATA_PATH=

HEAD_COUNT=1
DATA_COUNT=-1
PRETRAINED_PATH=


MODEL_NAME=

NUM_EPOCHS=2
LR=3e-6
LR_SCHE=cosine
EVAL_STEPS=25
SAVE_STEPS=25

EXP_NAME=
OUTPUT_DIR=
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    sft_decoder.py \
    --decoder_path ${PRETRAINED_PATH} \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --evaluation_strategy "no" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_only_model True \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 100 \
    --learning_rate ${LR} \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type ${LR_SCHE} \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
