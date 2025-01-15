# Semantics-Informed Group Interventions for Concept Bottleneck Models

An ETH Project for the [Deep Learning](https://da.inf.ethz.ch/teaching/2024/DeepLearning/) module in 2024.

## Contributors
Michał Mikuta (19-948-124), 
Mikael Makonnen (23-950-900),
Sari Issa (18-745-273) and
Max Buckley (22-906-853) 

## Datasets

This project uses the following three datasets:

*   [CUB (Birds dataset)](https://paperswithcode.com/dataset/cub-200-2011)
*   [CelebA (Celebrity dataset)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
*   [AwA2 (animals with attributes 2)](https://cvml.ista.ac.at/AwA2/)


# Data Processing

The datasets should be downloaded on first use and then reused for subsequent runs. if you run the respective file in `data/` it will download the respective dataset. Each of the corresponding data loader functions will return a dict of strings "TRAIN|VAL|TEST" mapped to a dataloader for that split of the data. The CUB dataset does not have a "VAL" split.

Warning: The [CelebA dataset is janky](https://github.com/pytorch/vision/issues/1920) and often returns a Google Drive error on attempting to fetch the large zip file programatically. As such you [should download that file manually](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) and extract it to `data/celeba/img_align_celeba/`.

Note: The AWA2 Dataset `AwA2-data.zip` is ~14Gb compressed.


# How to run

To train the CBM, run `data/celeba/img_align_celeba/`
