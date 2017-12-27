#!/usr/bin/env bash

# this is a training script
# defining global parameters
GPUS='0'
TRAIN_REC_PATH=./data/train.rec
VAL_REC_PATH=./data/val.rec
NETWORK=resnet50
BATCH_SIZE=16
DATA_SHAPE=512
PRETRAINED=./model/ssd_resnet50_512
OPTIMIZER=adam
LR=0.0004
TENSORBOARD=True
LR_STEPS=20,40,60
FREEZE=""
ROLLING=True
ROLLING_TIME=4
EPOCH=222
END_EPOCH=500
PREFIX="./output/exp3_fixed_params/ssd_"

python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE} \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --pretrained ${PRETRAINED} \
    --epoch ${EPOCH} \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch ${END_EPOCH}
