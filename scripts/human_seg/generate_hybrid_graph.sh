#!/usr/bin/env bash
# dataset: path to dataset. It is assumed that there are train and test folder
# in this path.
# portion: train or test. To generate the hybrid graph for test dataset select test
# To generate augmented hybrid graph for train select train
# hybrid_graphs: path to hybrid graph
python mesh2hybrid_converter.py \
--dataset ./dataset/human_seg \
--portion test \
--hybrid_graphs ./dataset/human_seg/hybrid_graphs
