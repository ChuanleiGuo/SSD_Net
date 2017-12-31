#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset kitti --set training --target $DIR/../data/kitti_train.lst --shuffle True --root $DIR/../data/KITTI
python $DIR/prepare_dataset.py --dataset kitti --set testing --target $DIR/../data/kitti_val.lst --shuffle False --root $DIR/../data/KITTI
