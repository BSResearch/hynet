#!/usr/bin/env bash
# Run the test. For more options please see the base_options.py and
# test_options.py
# dataroot: data root path. It has been assumed that there are train and test folders
# in this path
# results_dir: The trained model predicts the segmentation label for the corresponding
# and saves the segmented test data in this folder
# model_file: path to saved model for testing the test dataset
# save_prediction_for_test_files: If True the segmentation result of test data will be saved in result_dir
python test.py \
--dataroot ./datasets/human_seg/augmented_hybrid_graphs \
--batch_size 16 \
--results_dir ./datasets/human_seg/test_results \
--model_file ./checkpoints/human_seg_0/4_net.pth \
--classification_element edge \
--save_prediction_for_test_files True
