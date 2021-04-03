#!/usr/bin/env bash
# Run the training. For more option please see the base_options.py and
# train_options.py
# dataroot: dataset path. It has been assumed that there are train and test folder
# in this path.
#lr: learning rate
#name: A folder with this name will be added to ./checkpoints folder. The trained model will
# be saved in this folder
python train.py \
--dataroot ../../datasets/human_seg/augmented_hybrid_graphs \
--lr 0.003 \
--batch_size 16 \
--name human_seg_test \
--classification_element edge\
