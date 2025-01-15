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

The training of the CBM architecture has been performed on Google Collab with the following packages installed:
```
Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
torchvision == 0.20.1+cu121
numpy == 1.26.4
pandas == 2.2.2
torch == 2.5.1+cu121
requests == 2.32.3
tqdm == 4.67.1
```
To train the CBM, run `notebooks/ConceptBottleneckBirds.ipynb`. 
---
All following scripts may be run on the Euler cluster of ETHZ after having loaded the correct libraries:
```bash
  
```
For clustering, run `experiments/{DATASET}_clusters.py`. Note that one needs to install the `clustpy` packages to run clustering.

To train/test the realignment networks, it is easiest to call the `sbatch` command on SLURM:
```bash
  sbatch jobscript_cv_parallelized # train realignment networks on CUB dataset
  sbatch jobscript_test # test realignment networks on CUB dataset
```

To iterate over the maximum number of interventions, `sbatch` jobscripts are also provided:
```
  sbatch jobscript_maxinter # train realignment networks on CUB dataset with maximum number of interventions iteration
  sbatch jobscript_test_maxinter # test the aforementioned networks
  python3 realignment/maxinter_visualize.py # visualize the results
```
