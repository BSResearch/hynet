#!/usr/bin/env bash
# dataset: path to dataset. It is assumed that there are train and test folder
# in this path.
# portion: train or test. To generate the hybrid graph for test dataset select test
# To generate augmented hybrid graph for train select train
# hybrid_graphs: path to hybrid graph
# jitter_mean: mean of jitter, used for augmentation for train data
# jitter_var: variance of jitter, used for augmentation for train data.
# For human segmentation dataset, the proper value to keep the shape of the object is 0.002
#slide_vert_percentage, used for augmentation for train data
# augmentation does not apply on test data. 
python mesh2hybrid_converter.py \
--dataset ./dataset/human_seg \
--portion test \
--hybrid_graphs ./dataset/human_seg/hybrid_graphs \
--jitter_mean 0.0 \
--jitter_var 0.002 \
--slide_vert_percentage 0.2
