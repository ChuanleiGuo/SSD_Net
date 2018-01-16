#!/usr/bin/env bash

# this is a training script
# defining global parameters
GPUS='0,1,2,3'
TRAIN_REC_PATH=./data/car_train.rec
VAL_REC_PATH=./data/car_val.rec
NETWORK=resnet101-rolling
BATCH_SIZE=8
DATA_SHAPE=2560 768
PRETRAINED=./model/resnet-101
OPTIMIZER=adam
LR=0.0004
TENSORBOARD=True
LR_STEPS=20,40,100,200
FREEZE="^(stage1_|stage2_).*"
EPOCH=0
ROLLING=True
ROLLING_TIME=4
END_EPOCH=400
PREFIX="./output/exp2/ssd_"

# python ./train.py \
#     --train-path ${TRAIN_REC_PATH} \
#     --val-path ${VAL_REC_PATH} \
#     --network ${NETWORK} \
#     --batch-size 128 \
#     --data-shape 768 2560 \
#     --gpus ${GPUS} \
#     --prefix ${PREFIX} \
#     --finetune 1 \
#     --optimizer ${OPTIMIZER} \
#     --tensorboard ${TENSORBOARD} \
#     --lr ${LR} \
#     --lr-steps ${LR_STEPS} \
#     --freeze '' \
#     --num-class 1 \
#     --class-names Car \
#     --overlap 0.7 \
#     --rolling ${ROLLING} \
#     --rolling_time ${ROLLING_TIME} \
#     --end-epoch 100


python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape 768 2560 \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --pretrained ${PRETRAINED} \
    --epoch ${EPOCH} \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze ${FREEZE} \
    --num-class 1 \
    --class-names Car \
    --overlap 0.7 \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch ${END_EPOCH}
