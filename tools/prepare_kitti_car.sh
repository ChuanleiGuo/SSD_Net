#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset kitti_car --set training --target $DIR/../data/car_train.lst --shuffle True --root $DIR/../data/KITTI
python $DIR/prepare_dataset.py --dataset kitti_car --set testing --target $DIR/../data/car_val.lst --shuffle False --root $DIR/../data/KITTI
