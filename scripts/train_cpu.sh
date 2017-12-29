#!/usr/bin/env bash

# this is a training script
# defining global parameters
# TRAIN_REC_PATH=./data/train.rec
# VAL_REC_PATH=./data/val.rec
TRAIN_REC_PATH=/Users/chuan/Developer/VOCdevkit/data/train.rec
VAL_REC_PATH=/Users/chuan/Developer/VOCdevkit/data/val.rec
NETWORK=resnet50-rolling
BATCH_SIZE=1
DATA_SHAPE=512
PRETRAINED=./model/ssd_resnet50_512
PREFIX=./model/ssd_
OPTIMIZER=adam
TENSORBOARD=True
LR_STEPS=20,40,60

python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE} \
    --gpus "" \
    --prefix ${PREFIX} \
    --finetune 222 \
    --epoch 222 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --rolling True \
    --rolling_time 4 \
    --end-epoch 223
