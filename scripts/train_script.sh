#!/usr/bin/env bash

# this is a training script
# defining global parameters
GPUS='0,1,3'
TRAIN_REC_PATH=./data/train.rec
VAL_REC_PATH=./data/val.rec
NETWORK=resnet50
BATCH_SIZE=32
DATA_SHAPE=512
PRETRAINED=./model/ssd_resnet50_512
OPTIMIZER=adam
TENSORBOARD=True
LR_STEPS=20,40,60
FREEZE=""
ROLLING=True
ROLLING_TIME=4
END_EPOCH=1200
PREFIX="./output/exp2/ssd_"

python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE} \
    --gpus ${GPUS} \
    --pretrained ${PRETRAINED} \
    --prefix ${PREFIX} \
    --epoch 222 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr-steps ${LR_STEPS} \
    --freeze ${FREEZE} \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch ${END_EPOCH}
