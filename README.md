# HyNet
HyNet: 3D Segmentation Using Hybrid Graph Networks

HyNet is a novel representation learning framework that encodes mesh elements by focusing on the most relevant parts of geometric structure. We have adopted HyNet for 3D mesh segmentation.

![overview5](https://user-images.githubusercontent.com/81344957/112779416-af510400-9014-11eb-9362-912ccf6687b4.jpg)

## Installation

1. Clone this repository.
2. Install dependencies using environment.yml .

## 3D Shape Segmentation
Run the related bash script from ./scripts for the following tasks. For segmentation on human dataset we have:

- Download the dataset using following link:  
https://figshare.com/s/0587ec730eac72b41aa7

- Convert mesh files to hybrid graph
```
./scripts/human_seg/generate_hybrid_graph.sh
```
- To train a HyNet model,
```
./scripts/human_seg/train.sh
```
- Run test and save predicted segmentation
```
./scripts/human_seg/test.sh
```
- Visualize segmentation results:
```
./scripts/human_seg/mesh_viewer.sh
```
- Get pretrained model

## Results
![RESULTS](https://user-images.githubusercontent.com/81344957/112779505-de677580-9014-11eb-922a-d3c50cc397dd.jpg)

