# Semantics-Informed Group Interventions for Concept Bottleneck Models

---

## Contributors
- Michał Mikuta 
- Mikael Makonnen  
- Sari Issa
- Max Buckley

---

## Datasets
This project uses the following three datasets:
- [CUB (Birds dataset)](https://paperswithcode.com/dataset/cub-200-2011)  
- [CelebA (Celebrity dataset)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
- [AwA2 (Animals with attributes 2)](https://cvml.ista.ac.at/AwA2/)

---

## Data Processing
- The datasets should be downloaded on first use and then reused for subsequent runs. If you run the respective file in `data/`, it will download the respective dataset.  
- Each of the corresponding data loader functions will return a dict of strings `"TRAIN|VAL|TEST"` mapped to a dataloader for that split of the data.  
- **Note:** The CUB dataset does not have a `"VAL"` split.

> **Warning**  
> The [CelebA dataset is janky](https://github.com/pytorch/vision/issues/1920) and often returns a Google Drive error when attempting to fetch the large zip file programmatically. As such, you [should download that file manually](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) and extract it to `data/celeba/img_align_celeba/`.

> **Note**  
> The AWA2 Dataset `AwA2-data.zip` is ~14Gb compressed.

---

## How to Run

### Steps to Reproduce
1. **Download Data**  
2. **Cluster Data**  
3. **Train Models**  
4. **Perform Final Visualizations**

### Training Environment CBM
The training of the CBM architecture has been performed on Google Collab with the following packages installed:

```plaintext
Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
torchvision == 0.20.1+cu121
numpy == 1.26.4
pandas == 2.2.2
torch == 2.5.1+cu121
requests == 2.32.3
tqdm == 4.67.1
