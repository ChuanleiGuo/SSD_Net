#!/usr/bin/env bash

# this is a training script
# defining global parameters
GPUS='1,2,4'
TRAIN_REC_PATH=./data/train.rec
VAL_REC_PATH=./data/val.rec
NETWORK=resnet50-rb
BATCH_SIZE=32
DATA_SHAPE=512
PRETRAINED=./model/ssd_resnet50_512
OPTIMIZER=SGD
LR=0.004
TENSORBOARD=True
LR_STEPS=20,40,60,100,200,400,600,800
FREEZE=""
ROLLING=True
ROLLING_TIME=4
END_EPOCH=1000
PREFIX="./output/exp1/ssd_"

python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size 128 \
    --data-shape ${DATA_SHAPE} \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --finetune 1 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --monitor 40 \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch 5


python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape ${DATA_SHAPE} \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --resume 5 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --monitor 40 \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch ${END_EPOCH}
