# Semantics-Informed Group Interventions for Concept Bottleneck Models

An ETH Project for the [Deep Learning](https://da.inf.ethz.ch/teaching/2024/DeepLearning/) module in 2024.

## Datasets

This project uses the following three datasets:

*   [CUB (Birds dataset)](https://paperswithcode.com/dataset/cub-200-2011)
*   [CelebA (Celebrity dataset)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
*   [AwA2 (animals with attributes 2)](https://cvml.ista.ac.at/AwA2/)
*   

# Data Processing

The datasets should be downloaded on first use and then reused for subsequent runs. if you run the respective file in `data/` it will download the respective dataset. Each of the corresponding data loader functions will return a dict of strings "TRAIN|VAL|TEST" mapped to a dataloader for that split of the data. The CUB dataset does not have a "VAL" split.

Warning: The [CelebA dataset is janky](https://github.com/pytorch/vision/issues/1920) and often returns a Google Drive error on attempting to fetch the large zip file. As such you should download that file manually and extract it to `data/celeba/img_align_celeba/`.