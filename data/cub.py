import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data import utils

import os
import pandas as pd
import requests
from tqdm import tqdm
import tarfile

import multiprocessing as mp

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


class CUBDataset(Dataset):
    """
    Create a PyTorch dataset from a list of image paths.

    Args:
        image_paths: List of paths to image files
        transform: Optional transform to be applied on images
                  (if None, will convert to tensor and normalize)
    """

    def __init__(self, image_paths, concepts, labels, ids, transform=None, split="train"):
      self.concepts = []
      self.labels = []
      self.images = []
      self.ids = ids

      assert type(concepts) == type(labels) == type(image_paths) == list, (
        "concepts, labels, and image_paths must be of the same type, list. \nGot: %s, %s, %s" % (type(concepts), type(labels), type(image_paths)))

      assert len(image_paths) == len(concepts) == len(labels), (
        "Number of images, concepts, and labels must match")

      base_transforms = transforms.Compose(
      [
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      
      # Default transform if none provided
      self.transform = transform if transform is not None else transforms.Compose([])

      for image_path, concept, label in zip(image_paths, concepts, labels):
        try:
          image = Image.open(image_path).convert('RGB')
        except Exception as e:
          print(f"Error loading image {image_path}: {str(e)}")
        # Apply base transforms
        image = base_transforms(image)
        self.images.append(image)
        self.concepts.append(self._convert_concepts_to_tensor(concept))
        self.labels.append(torch.tensor(label, dtype=torch.long))

      # write the concept Tensor out as a Pandas dataframe
      concept_arrays = [t.numpy() if torch.is_tensor(t) else np.array(t) for t in self.concepts]
      # Stack all arrays into a single 2D array
      concepts = np.vstack(concept_arrays)
      #concept_names = None
      concept_names = utils.read_txt_file(
          "data/cub/attributes.txt", 2, ["id", "concept_names"])["concept_names"]
      print(concept_names)
      if concept_names is None:
          concept_names = [f'concept_{i}' for i in range(concepts.shape[1])]
      # Create DataFrame with IDs and features
      df = pd.DataFrame(concepts, columns=concept_names)
      df.insert(0, 'id', ids)  # Add IDs as the first column
      # Save to CSV
      df.to_csv("data/cub/output/concepts_%s.csv" % split, index=False)
      print(f"Saved CSV file to: data/cub/output/concepts_%s.csv" % split)



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
        # Load image
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        concept = self.concepts[idx]

        return image, concept, label




def download_cub200_2011():
    """
    Downloads the CUB-200-2011 dataset and extracts it.
    Returns the path to the extracted dataset.
    """
    # Create a directory for the dataset
    base_dir = './data/cub/'
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

def read_txt_file(filepath, num_cols):
    """
    Safely read space-separated text files with a specific number of columns.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= num_cols:
                data.append(parts[:num_cols])
    return pd.DataFrame(data)

def load_cub_data(data_dir):
    """
    Loads and organizes the CUB dataset metadata.
    Returns dictionaries for image paths, labels, and attribute data.
    """
    # Load image paths and labels using the safe reader
    images_df = read_txt_file(os.path.join(data_dir, 'images.txt'), 2)
    images_df.columns = ['image_id', 'image_path']
    images_df['image_id'] = images_df['image_id'].astype(int)

    labels_df = read_txt_file(os.path.join(data_dir, 'image_class_labels.txt'), 2)
    labels_df.columns = ['image_id', 'class_id']
    labels_df['image_id'] = labels_df['image_id'].astype(int)
    labels_df['class_id'] = labels_df['class_id'].astype(int)

    # Load train/test split
    train_test_df = read_txt_file(os.path.join(data_dir, 'train_test_split.txt'), 2)
    train_test_df.columns = ['image_id', 'is_training']
    train_test_df['image_id'] = train_test_df['image_id'].astype(int)
    train_test_df['is_training'] = train_test_df['is_training'].astype(int)

    # Load attributes using the safe reader
    attr_df = read_txt_file(os.path.join(data_dir, 'attributes/image_attribute_labels.txt'), 5)
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

def get_data_dict():
  data_dir = download_cub200_2011()
  data = load_cub_data(data_dir)
  return data


def split_train_split(data_dir):
    # Initialize the split dictionary
    # Load train/test split
    train_test_df = read_txt_file(os.path.join(data_dir, 'CUB_200_2011', 'train_test_split.txt'), 2)
    train_test_df.columns = ['image_id', 'is_training']
    train_test_df['image_id'] = train_test_df['image_id'].astype(int)
    train_test_df['is_training'] = train_test_df['is_training'].astype(int)

    splits = {}
    
    train_df = train_test_df[train_test_df['is_training'] == 1]


    # Get indices where value in dict is 1 (training)
    train_ids = [id for id in train_df['image_id'].values]

    # Randomly shuffle these indices
    np.random.seed(42)
    shuffled_indices = np.random.permutation(train_ids)

    # Calculate split point for 80/20 split of training data
    n_train = int(len(train_ids) * 0.8)

    # First set all indices in original dict to 'test'
    for idx in train_test_df['image_id']:
        splits[idx] = 'test'

    # Update training indices
    for idx in shuffled_indices[:n_train]:
        splits[idx] = 'train'

    # Update validation indices
    for idx in shuffled_indices[n_train:]:
        #print(idx)
        splits[idx] = 'val'
    with open(os.path.join(data_dir, 'output', 'train_val_test_split.txt'), 'w') as f:
        for id_, split in splits.items():
            f.write(f'{id_}\t{split}\n')
    return splits

def get_train_val_test_datasets(data):

  splits = split_train_split("data/cub/")
  data['split'] = splits

  # First get sorted IDs for train and test
  train_ids = sorted([id for id, split in data['split'].items() if split == "train"])
  val_ids = sorted([id for id, split in data['split'].items() if split == "val"])
  test_ids = sorted([id for id, split in data['split'].items() if split == "test"])

  print("Train IDs = %d" % len(train_ids))
  print("Val IDs = %d" % len(val_ids))
  print("Test IDs = %d" % len(test_ids))

  train_transforms = transforms.Compose([])

  val_transforms = transforms.Compose([])

  test_transforms = transforms.Compose([])

  # Create training dataset using the sorted train IDs
  train_dataset = CUBDataset(
      image_paths=[data['image_paths'][id] for id in train_ids],
      concepts=[data['attributes'][id] for id in train_ids],
      labels=[data['labels'][id] for id in train_ids],
      ids=train_ids,
      transform=train_transforms,
      split="train"
  )

  val_dataset = CUBDataset(
      image_paths=[data['image_paths'][id] for id in val_ids],
      concepts=[data['attributes'][id] for id in val_ids],
      labels=[data['labels'][id] for id in val_ids],
      ids=val_ids,
      transform=val_transforms,
      split="val"
  )

  # Create validation dataset using the sorted test IDs
  test_dataset = CUBDataset(
      image_paths=[data['image_paths'][id] for id in test_ids],
      concepts=[data['attributes'][id] for id in test_ids],
      labels=[data['labels'][id] for id in test_ids],
      ids=test_ids,
      transform=test_transforms,
      split="test"
  )
   # Verify the split
  print(f"Training samples: {len(train_dataset)}")
  print(f"Validation samples: {len(val_dataset)}")
  print(f"Test samples: {len(test_dataset)}")

  return train_dataset, val_dataset, test_dataset



def get_train_val_test_loaders(train_dataset, val_dataset, test_dataset, batch_size):
  num_cpus = mp.cpu_count()
  num_workers = num_cpus - 2
  print(f"Number of CPUs: {num_cpus}")
  print(f"Number of workers: {num_workers}")

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,  # No need to shuffle validation data
      num_workers=num_workers
  )
  return train_loader, val_loader, test_loader


if __name__ == "__main__":    
    data_dict = get_data_dict()
    print("Creating Datasets")
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(data_dict)
    print("Creating Dataloaders")
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=512)