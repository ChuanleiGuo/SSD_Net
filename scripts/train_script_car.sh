#!/usr/bin/env bash

# this is a training script
# defining global parameters
GPUS='0,1,2,3'
TRAIN_REC_PATH=./data/car_train.rec
VAL_REC_PATH=./data/car_val.rec
NETWORK=resnet50-rolling
BATCH_SIZE=32
DATA_SHAPE=2560 768
PRETRAINED=./model/ssd_resnet50_2560
OPTIMIZER=adam
LR=0.0001
TENSORBOARD=True
LR_STEPS=20,40,100,200,400
FREEZE=""
ROLLING=True
ROLLING_TIME=4
END_EPOCH=500
PREFIX="./output/exp1/ssd_"

python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size 128 \
    --data-shape 2560 768 \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --finetune 1 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --num-class 1 \
    --class-names Car \
    --overlap 0.7 \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch 100


python ./train.py \
    --train-path ${TRAIN_REC_PATH} \
    --val-path ${VAL_REC_PATH} \
    --network ${NETWORK} \
    --batch-size ${BATCH_SIZE} \
    --data-shape 2560 768 \
    --gpus ${GPUS} \
    --prefix ${PREFIX} \
    --resume 100 \
    --optimizer ${OPTIMIZER} \
    --tensorboard ${TENSORBOARD} \
    --lr ${LR} \
    --lr-steps ${LR_STEPS} \
    --freeze '' \
    --num-class 1 \
    --class-names Car \
    --overlap 0.7 \
    --rolling ${ROLLING} \
    --rolling_time ${ROLLING_TIME} \
    --end-epoch ${END_EPOCH}
