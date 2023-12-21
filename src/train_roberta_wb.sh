#!/usr/bin/env bash


MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext-large
VERSION=v5
MAX_NUM_QUESTIONS=8
DATASET="weibo"

MAX_SEQ1_LENGTH=110
MAX_SEQ2_LENGTH=10
CAND_K=3
LAMBDA=${1:-0.5}
PRIOR=${2:-rand}
MASK=${3:-0.0}
echo "lambda = $LAMBDA, prior = $PRIOR, mask = $MASK"

DATA_DIR=$PJ_HOME/data/${DATASET}/
OUTPUT_DIR=$PJ_HOME/output/${DATASET}_${MODEL_NAME_OR_PATH}/${DATASET}_${MODEL_NAME_OR_PATH}_${PRIOR}_l${LAMBDA}
NUM_TRAIN_EPOCH=8
GRADIENT_ACCUMULATION_STEPS=2
PER_GPU_TRAIN_BATCH_SIZE=8
PER_GPU_TEST_BATCH_SIZE=16

LOGGING_STEPS=200
SAVE_STEPS=200

HS=256
SHARE_HS=128

python3 train.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type ${MODEL_TYPE} \
  --max_seq1_length ${MAX_SEQ1_LENGTH} \
  --max_seq2_length ${MAX_SEQ2_LENGTH} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --do_train \
  --do_test \
  --evaluate_during_training \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --num_train_epochs ${NUM_TRAIN_EPOCH} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
  --per_gpu_test_batch_size ${PER_GPU_TEST_BATCH_SIZE} \
  --logging_steps ${LOGGING_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --logic_lambda ${LAMBDA} \
  --prior ${PRIOR} \
  --overwrite_output_dir \
  --temperature 1.0 \
  --hs ${HS} \
  --share_hs ${SHARE_HS}

