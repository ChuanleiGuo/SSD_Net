#!/usr/bin/env bash
# This is a training script
# defining global parameters
GPUS='0,1,3'
REC_PATH=./data/val.rec
NETWORK=resnet50
BATCH_SIZE=64
DATA_SHAPE=512

python ./evaluate.py \
    --rec-path ${REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE} \
    --gpus ${GPUS} \
    --epoch 222 

