export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PER_DEVICES_BS=2
GRAD_ACC=64

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
ACCELERATE_CONFIG_PATH=accelerate_config/gpu${GPU_COUNT}_zero3_acc${GRAD_ACC}.yaml


REPORT_TO=wandb
PORT=38900

export WANDB__SERVICE_WAIT=600
export WANDB_MODE=online
NUM_EPOCHS=1

export MAX_COL=
export MAX_ROW=
DATA_PATH=
EVAL_DATA_PATH=

HEAD_COUNT=1

LR=5e-5
WARMUP_RATIO=0.05
EVAL_STEPS=40
SAVE_STEPS=40
DATA_COUNT=-1
LR_SCHE=cosine_with_min_lr
LR_SCHE_ARGS='{"min_lr_rate":0.02}'



BASE_MODEL=
ENCODER_PATH=
MODEL_NAME=
NORM=not
EXP_NAME=
OUTPUT_DIR=
mkdir -p ${OUTPUT_DIR}
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_codellama_qformer.py \
    --decoder_path ${BASE_MODEL} \
    --encoder_path $ENCODER_PATH \
    --projector_num_heads ${HEAD_COUNT} \
    --encoder_hidden_size 4096 \
    --decoder_hidden_size 4096 \
    --data_path $DATA_PATH \
    --data_count $DATA_COUNT \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ${OUTPUT_DIR} \
    --freeze_projector False \
    --freeze_encoder False \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --evaluation_strategy "no" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHE} \
    --lr_scheduler_kwargs ${LR_SCHE_ARGS} \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained False \
    --torch_dtype float32
