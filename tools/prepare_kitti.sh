#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset kitti --set taining --target $DIR/../data/train.lst --shuffle True --root $DIR/../data/kitti
python $DIR/prepare_dataset.py --dataset kitti --set testing --target $DIR/../data/val.lst --shuffle False --root $DIR/../data/kitti
