#!/usr/bin/env bash
# The mesh viewer gets the hybrid graph ( --input_graph_name arg from --input_folder arg)
# and view the segmented mesh data. We provide an example of segmentation result
# in test_results folder.
# you can use the mesh viewer to visualize the ground truth or prediction given as the arg of --mode
# example: --mode prediction or --mode gt
# Use screenshot_folder_to_save to give a path to save the screenshots of the mesh.
# The screenshots will be saved with the graph name in the screenshot folder.
python mesh_viewer.py \
--input_graph_name shrec__9_0_nef_graph.bin \
--input_folder ./scripts/human_seg/test_results \
--mode prediction \
--screenshot_folder_to_save ./scripts/human_seg/test_results/screenshots
