import os
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader


import os
import time
import requests
from torchvision.datasets import CelebA
from torchvision import transforms

def download_file_from_google_drive(file_id, destination, filename, max_retries=3, retry_delay=3600):
    """
    Downloads a file from Google Drive with retry logic
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            
            token = get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            save_response_content(response, destination)
            print(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            if "Too many users" in str(e):
                remaining_attempts = max_retries - attempt - 1
                if remaining_attempts > 0:
                    print(f"\nGoogle Drive rate limit hit for {filename}.")
                    print(f"Waiting {retry_delay} seconds before retry.")
                    print(f"Remaining attempts: {remaining_attempts}")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to download {filename} after {max_retries} attempts")
                    return False
            else:
                print(f"Unexpected error downloading {filename}: {e}")
                return False
    
    return False

def download_celeba_text_files(root_dir='./celeba', max_retries=3, retry_delay=3600):
    """
    Downloads all necessary CelebA files
    """
    # Create directories
    os.makedirs(root_dir, exist_ok=True)
    
    # Define all required files and their Google Drive IDs
    files = {
        'img_align_celeba.zip': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
        'list_attr_celeba.txt': '0B7EVK8r0v71pblRyaVFSWGxPY0U',
        'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
        'list_bbox_celeba.txt': '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
        'list_landmarks_align_celeba.txt': '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
        'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
    }
    
    print("Starting CelebA text file download...")
    
    # Download each file
    for filename, file_id in files.items():
        destination = os.path.join(root_dir, filename)
        
        if os.path.exists(destination):
            print(f"{filename} already exists, skipping...")
            continue
            
        print(f"\nDownloading {filename}...")
        success = download_file_from_google_drive(
            file_id=file_id,
            destination=destination,
            filename=filename,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if not success:
            print(f"\nManual download instructions for {filename}:")
            print(f"1. Visit: https://drive.google.com/uc?id={file_id}")
            print(f"2. Download and place in: {destination}")
    
    print("\nDownload process completed!")
    print("\nIf any files failed to download automatically, you can download them manually:")
    print("Main dataset page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    for filename, file_id in files.items():
        print(f"\n{filename}:")
        print(f"Google Drive link: https://drive.google.com/uc?id={file_id}")
        print(f"Save to: {os.path.join(root_dir, filename)}")

def check_and_download_celeba(root_dir='./', download=True):
    """
    Checks if CelebA dataset exists and downloads it if not present.
    
    Args:
        root_dir (str): Directory where the dataset should be stored
        download (bool): Whether to download the dataset if not found
    
    Returns:
        dataset: The CelebA dataset object
        bool: Whether the dataset was already present
    """
    # Create the root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    download_celeba_text_files()
    
    # Check if the dataset files exist
    celeba_dir = os.path.join(root_dir, 'celeba')
    img_dir = os.path.join(celeba_dir, 'img_align_celeba')
    
    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
        print("CelebA dataset found at", celeba_dir)
        return
    else:
        
        print("CelebA dataset not found. Downloading...")
        # Initialize the dataset with download=True
        _ = CelebA(root=root_dir, 
                        split='all',
                        transform=None,
                        download=True)
        print("Dataset downloaded successfully!")


def get_train_val_test_loaders(batch_size=32):
    """
    Creates DataLoaders for the train/val/test splits of the CelebA dataset
    
    Args:
        batch_size (int): Size of each batch
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    check_and_download_celeba()

    root_dir='./'
    num_workers=4
    
        # Define standard transforms
    transform = transforms.Compose([
        transforms.Resize((218, 178)),
        transforms.CenterCrop((178, 178)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CelebA(root=root_dir,
                    split='train',
                    transform=transform,
                    download=False)
    val_dataset = CelebA(root=root_dir,
                split='valid',
                transform=transform,
                download=False)
    test_dataset = CelebA(root=root_dir,
                split='test',
                transform=transform,
                download=False)   
        
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle only training data
        num_workers=num_workers
        )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle only training data
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle only training data
        num_workers=num_workers
    )

    return {"TRAIN": train_dataloader, "VAL": val_dataloader, "TEST": test_dataloader}


if __name__ == "__main__":
    # Create dataloaders
    dataloaders = get_train_val_test_loaders(32)
    
    for key, dataloader in dataloaders.items():
        print(f"Total samples in {key}: {len(dataloader.dataset)}")

    # Test the train dataloader
    for images, labels in dataloaders["TRAIN"]:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break