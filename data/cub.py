import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils

import os
import pandas as pd
import requests
from tqdm import tqdm
import tarfile

from PIL import Image


def download_cub200_2011():
    """
    Downloads the CUB-200-2011 dataset and extracts it.
    Returns the path to the extracted dataset.
    """
    # Create a directory for the dataset
    base_dir = './cub/'
    dataset_dir = os.path.join(base_dir, 'CUB_200_2011')

    # Check if dataset already exists
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'images.txt')):
        print("Dataset already downloaded and extracted.")
        return dataset_dir

    os.makedirs(base_dir, exist_ok=True)

    # URL for the dataset
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    tgz_path = os.path.join(base_dir, 'CUB_200_2011.tgz')

    # Download only if not already downloaded
    if not os.path.exists(tgz_path):
        print("Downloading CUB-200-2011 dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(tgz_path, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024),
                            total=total_size//1024,
                            unit='KB'):
                f.write(data)

    # Extract only if not already extracted
    if not os.path.exists(dataset_dir):
        print("\nExtracting dataset...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(base_dir)

    # Remove the downloaded tar file to save space
    if os.path.exists(tgz_path):
        os.remove(tgz_path)

    return dataset_dir


def load_cub_data(data_dir):
    """
    Loads and organizes the CUB dataset metadata.
    Returns dictionaries for image paths, labels, and attribute data.
    """
    # Load image paths and labels using the safe reader
    images_df = utils.read_txt_file(os.path.join(data_dir, 'images.txt'), 2)
    images_df.columns = ['image_id', 'image_path']
    images_df['image_id'] = images_df['image_id'].astype(int)

    labels_df = utils.read_txt_file(os.path.join(data_dir, 'image_class_labels.txt'), 2)
    labels_df.columns = ['image_id', 'class_id']
    labels_df['image_id'] = labels_df['image_id'].astype(int)
    labels_df['class_id'] = labels_df['class_id'].astype(int)

    # Load train/test split
    train_test_df = utils.read_txt_file(os.path.join(data_dir, 'train_test_split.txt'), 2)
    train_test_df.columns = ['image_id', 'is_training']
    train_test_df['image_id'] = train_test_df['image_id'].astype(int)
    train_test_df['is_training'] = train_test_df['is_training'].astype(int)

    # Load attributes using the safe reader
    attr_df = utils.read_txt_file(os.path.join(data_dir, 'attributes/image_attribute_labels.txt'), 5)
    attr_df.columns = ['image_id', 'attribute_id', 'is_present', 'certainty', 'time']
    attr_df = attr_df.astype({
        'image_id': int,
        'attribute_id': int,
        'is_present': int,
        'certainty': int,
        'time': float
    })

    print("Merging")

    # Merge dataframes
    data = images_df.merge(labels_df, on='image_id')
    data = data.merge(train_test_df, on='image_id')

    print("Creating Dictionaries")
    # Create dictionaries
    image_paths = {row['image_id']: os.path.join(data_dir, 'images', row['image_path'])
                  for _, row in data.iterrows()}

    labels = {row['image_id']: row['class_id'] - 1  # Convert to 0-based indexing
             for _, row in data.iterrows()}

    train_test = {row['image_id']: row['is_training']
                  for _, row in data.iterrows()}

    # Organize attributes
    print("Organizing Attributes")
    # This is the slow part. Optimize...
    attributes = {}
    for _, row in attr_df.iterrows():
        image_id = row['image_id']
        if image_id not in attributes:
            attributes[image_id] = []
        attributes[image_id].append({
            'attribute_id': row['attribute_id'],
            'is_present': row['is_present'],
            'certainty': row['certainty']
        })

    return {
        'image_paths': image_paths,
        'labels': labels,
        'train_test_split': train_test,
        'attributes': attributes
    }
    

class BirdsDataset(Dataset):
    """
    Create a PyTorch dataset from a list of image paths.

    Args:
        image_paths: List of paths to image files
        transform: Optional transform to be applied on images
                  (if None, will convert to tensor and normalize)
    """

    def __init__(self, image_ids, image_paths, concepts, labels, transform=None, save_concepts=False):
      self.image_ids = image_ids
      self.concepts = []
      self.labels = []
      self.images = []

      assert type(image_ids) == type(concepts) == type(labels) == type(image_paths) == list, (
        "concepts, labels, and image_paths must be of the same type, list. \nGot: %s, %s, %s" % (type(concepts), type(labels), type(image_paths)))

      assert len(image_ids) == len(image_paths) == len(concepts) == len(labels), (
        "Number of images, concepts, and labels must match")

      # Default transform if none provided
      self.transform = transform if transform is not None else transforms.Compose([
          transforms.Resize((224, 224)),  # Resize to standard size
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                            std=[0.229, 0.224, 0.225])
      ])

      for image_path, concept, label in zip(image_paths, concepts, labels):
        try:
          image = Image.open(image_path).convert('RGB')
        except Exception as e:
          print(f"Error loading image {image_path}: {str(e)}")



        # Apply transforms
        if self.transform:
          image = self.transform(image)
        self.images.append(image)

        self.concepts.append(self._convert_concepts_to_tensor(concept))
        self.labels.append(torch.tensor(label, dtype=torch.long))

      if save_concepts:
        # write the concept Tensor out as a Pandas dataframe
        concept_arrays = [t.numpy() if torch.is_tensor(t) else np.array(t) for t in self.concepts]

        # Stack all arrays into a single 2D array
        concepts = np.vstack(concept_arrays)
        #concept_names = None
        concept_names = utils.read_txt_file("cub/attributes.txt", 2, ["id","concept_names"])["concept_names"]
        print(concept_names)
        if concept_names is None:
            concept_names = [f'concept_{i}' for i in range(concepts.shape[1])]

        # Create DataFrame with IDs and features
        df = pd.DataFrame(concepts, columns=concept_names)
        print(f"DataFrame shape: {df.shape}")
        print(f"image_ids len: {len(image_ids)}")

        df.insert(0, 'id', image_ids)  # Add IDs as the first column
        # Save to CSV
        print(f"DataFrame shape: {df.shape}")
        df.to_csv("cub/output/concepts_train.csv", index=False)
        print(f"Saved CSV file to: cub/output/concepts_train.csv")
        
            
    def _convert_concepts_to_tensor(self, concept_list):
        """
        Convert list of concept dictionaries to binary tensor.
        We use is_present field to create a binary vector.
        """
        # Create tensor of zeros
        concept_tensor = torch.zeros(312)

        # Fill in the binary values from is_present
        for i, concept_dict in enumerate(concept_list):
            concept_tensor[i] = 1.0 if concept_dict['is_present'] == 1.0 else 0.0

        return concept_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #idx = idx + 1
        # Load image
        image = self.images[idx]

        label = self.labels[idx]
        concept = self.concepts[idx]

        return image, concept, label



def get_train_test_loaders(batch_size):
    data_dir = download_cub200_2011()
    print(f"\nDataset directory: {data_dir}")

    print("\nLoading dataset metadata...")
    data = load_cub_data(data_dir)


    num_classes = len(set(data['labels'].values()))

    first_image_id = list(data['image_paths'].keys())[0]
    num_concepts = len(data['attributes'][first_image_id])

    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total number of images: {len(data['image_paths'])}")
    print(f"Number of training images: {sum(data['train_test_split'].values())}")

    # a map of int id to class label 0 train, 1 test
    print(f"Number of test images: {len(data['train_test_split']) - sum(data['train_test_split'].values())}")
    print(f"Number of classes: {num_classes}")

    # Example of accessing data for first image
    print(f"\nExample data for image {first_image_id}:")
    print(f"Image path: {data['image_paths'][first_image_id]}")
    print(f"Class label: {data['labels'][first_image_id]}")
    print(f"Is training: {data['train_test_split'][first_image_id]}")
    print(f"Number of concepts: {num_concepts}")


    # First get sorted IDs for train and test
    train_ids = sorted([id for id, is_train in data['train_test_split'].items() if is_train == 1])
    test_ids = sorted([id for id, is_train in data['train_test_split'].items() if is_train == 0])

    train_concepts = [data['attributes'][id] for id in train_ids]
    
    print("len(train_concepts) = %s" % len(train_concepts))
    #print("train_concepts = %s" % train_concepts)
    # These need the corresponding train_ids and likely the concpet mapped

    # Create training dataset using the sorted train IDs
    train_dataset = BirdsDataset(
        image_ids = [id for id in train_ids],
        image_paths=[data['image_paths'][id] for id in train_ids],
        concepts=train_concepts,
        labels=[data['labels'][id] for id in train_ids],
        save_concepts=True
    )
    

    # Create validation dataset using the sorted test IDs
    test_dataset = BirdsDataset(
        image_ids = [id for id in test_ids],
        image_paths=[data['image_paths'][id] for id in test_ids],
        concepts=[data['attributes'][id] for id in test_ids],
        labels=[data['labels'][id] for id in test_ids]
    )
    
    
    # Verify the split
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("Creating DataLoaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=4
    )

    return {"TRAIN": train_loader, "TEST": test_loader}


if __name__ == "__main__":
    dataloaders = get_train_test_loaders(batch_size=1024)
    #utils.dataloader_to_csv(dataloaders["Train"], "/output/cub_train.csv", column_names=[])