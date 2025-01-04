import os
# import torch
# import requests
import zipfile
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity
from PIL import Image
from torch.utils.data import DataLoader

import utils

class AWA2Dataset(VisionDataset):
    """
    Animals with Attributes 2 (AWA2) Dataset loader with caching functionality
    """
    
    base_url = "https://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    filename = "AwA2-data.zip"
    md5 = "08b3f5e4e2e9848468165"  # Replace with actual MD5
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(AWA2Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.train = train
        self.root = Path(root)
        self.data_dir = self.root / "Animals_with_Attributes2"
        
        #if download:
        #    self._download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it')
            
        # Load the dataset
        self._load_dataset()
    
    def _check_exists(self):
        """Check if the dataset exists in the root directory"""
        print(self.data_dir)
        return self.data_dir.exists()
    
    def _download(self):
        """Download and extract the dataset if it doesn't exist"""
        if self._check_exists():
            print('Dataset already downloaded and verified')
            return
        
        os.makedirs(self.root, exist_ok=True)
        
        try:
            print('Downloading AWA2 dataset...')
            download_url(self.base_url, self.root, self.filename, self.md5)
            
            print('Extracting downloaded file...')
            with zipfile.ZipFile(self.root / self.filename, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            
            # Clean up the zip file
            os.remove(self.root / self.filename)
            
            print('Dataset successfully downloaded and extracted')
            
        except Exception as e:
            # Clean up partially downloaded files on failure
            #if os.path.exists(self.root / self.filename):
            #    os.remove(self.root / self.filename)
            raise RuntimeError(f'Error downloading/extracting dataset: {str(e)}')
    
    def _load_dataset(self):
        """Load the dataset into memory"""
        # Load class information and splits
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.targets = []
        
        # Load train/test split information
        # You might need to adjust this based on the actual structure of AWA2
        split_file = 'classes.txt'# if self.train else 'testclasses.txt'
        split_path = self.data_dir / split_file
        
        # Load image paths and labels
        with open(split_path, 'r') as f:
            for line in f:
                print("Line = %s" % line)
                img_path, label = line.strip().split()
                self.images.append(self.data_dir / 'images' / img_path)
                self.targets.append(int(label))
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.images[index], self.targets[index]
        
        # Load the image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def __len__(self):
        return len(self.images)

# Example usage:
def get_train_test_loaders(batch_size=32):
    """
    Helper function to load the AWA2 dataset
    
    Args:
        root (str): Root directory where the dataset should be stored
        train (bool): If True, creates dataset from training set, otherwise from test set
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version
        download (bool): If True, downloads the dataset from the internet and puts it in root directory.
                        If dataset is already downloaded, it is not downloaded again.
    
    Returns:
        AWA2Dataset: The loaded dataset
    """
    root="./awa2" 
    transform=None 
    download=False
    
    num_workers=4
    train_dataset =  AWA2Dataset(
        root=root,
        train=True,
        transform=transform,
        download=download
    )
    test_dataset =  AWA2Dataset(
        root=root,
        train=False,
        transform=transform,
        download=download
    )
    
        # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle only training data
        num_workers=num_workers
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle only training data
        num_workers=num_workers
    )

    return {"TRAIN": train_dataloader,
            "TEST": test_dataloader}
    
    
def prepare_binary_concept_matrix():    
    # write the concept Tensor out as a Pandas dataframe
    #concept_arrays = [t.numpy() if torch.is_tensor(t) else np.array(t) for t in self.concepts]

    # Stack all arrays into a single 2D array
    #concepts = np.vstack(concept_arrays)
    #concept_names = None
    concept_names = list(utils.read_txt_file("data/awa2/Animals_with_Attributes2/predicates.txt", 2, ["id","concept"])["concept"].values)
    print(concept_names)
    print(type(concept_names))
    
    row_names = list(utils.read_txt_file("data/awa2/Animals_with_Attributes2/classes.txt", 2, ["id","animal"])["animal"].values)
    
    predicate_matrix_binary = utils.read_txt_file("data/awa2/Animals_with_Attributes2/predicate-matrix-binary.txt", len(concept_names), concept_names)
    predicate_matrix_binary.insert(0, 'animal', row_names)
    
    train_classes = list(utils.read_txt_file("data/awa2/Animals_with_Attributes2/trainclasses.txt", 1, ["animal"])["animal"].values)
    print(train_classes)

    predicate_matrix_binary = predicate_matrix_binary[predicate_matrix_binary['animal'].isin(train_classes)]

    print(predicate_matrix_binary)
    predicate_matrix_binary.to_csv("data/awa2/output/concepts_train.csv", index=False)

    """
    value_counts = splits['split'].value_counts()
    print(value_counts)
    
    filenames = splits[splits['split'] == "0"]['file_name'].tolist()
    print(filenames)
    print(len(filenames))
    
    concepts = utils.read_txt_file("data/celeba/list_attr_celeba.txt", 41)
    

    concepts = concepts.replace("-1", "0")
    print(concepts)
    print(len(concepts))
    
    filtered_concepts = concepts[concepts['file_name'].isin(filenames)]

    print(filtered_concepts)
    print(len(filtered_concepts))
    print(len(filenames))
    
    filtered_concepts.to_csv("data/celeba/output/concepts_train.csv", index=False)
    """
    return

    
    

if __name__ == "__main__":
    # Create dataloaders
    # Should take a batch size
    dataloaders = get_train_test_loaders(32)
    for key, dataloader in dataloaders.items():
        print(f"Total samples in {key}: {len(dataloader.dataset)}")

    # Test the train dataloader
    for images, labels in dataloaders["TRAIN"]:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    prepare_binary_concept_matrix()