"""
Dataset utility functions for downloading and checking dataset availability.

Datasets with automatic download (via torchvision):
  - MNIST, Fashion-MNIST, KMNIST, CIFAR-10, CIFAR-100, SVHN

Datasets requiring more setup:
  - Tiny-ImageNet: Requires manual download (see setup_tiny_imagenet)
  - UCI-Adult, Covertype: Cached via fetch_openml on first use
"""

import os
import sys
import shutil
from pathlib import Path


def ensure_data_dir(data_dir="./data"):
    """Create data directory if it doesn't exist."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def check_dataset_exists(dataset_name, data_dir="./data"):
    """
    Check if a dataset is already downloaded / ready.
    
    Returns: (exists: bool, status: str)
    """
    data_dir = ensure_data_dir(data_dir)
    name_lower = dataset_name.lower()
    
    # Datasets auto-downloaded by torchvision
    auto_download_datasets = {
        "mnist": os.path.join(data_dir, "MNIST", "raw"),
        "fashion-mnist": os.path.join(data_dir, "FashionMNIST", "raw"),
        "fashion_mnist": os.path.join(data_dir, "FashionMNIST", "raw"),
        "fashionmnist": os.path.join(data_dir, "FashionMNIST", "raw"),
        "kmnist": os.path.join(data_dir, "KMNIST", "raw"),
        "cifar10": os.path.join(data_dir, "cifar-10-batches-py"),
        "cifar100": os.path.join(data_dir, "cifar-100-python"),
        "svhn": os.path.join(data_dir, "svhn"),
    }
    
    # Special datasets
    if name_lower in ("tiny-imagenet", "tiny_imagenet", "tinyimagenet"):
        train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")
        exists = os.path.isdir(train_dir) and os.path.isdir(val_dir)
        if exists:
            return True, f"✓ Tiny-ImageNet found at {data_dir}/tiny-imagenet-200"
        else:
            return False, (
                f"✗ Tiny-ImageNet not found.\n"
                f"  Expected: {data_dir}/tiny-imagenet-200/{{train,val}}\n"
                f"  --> See setup_tiny_imagenet() for download instructions."
            )
    
    if name_lower in ("uci-adult", "adult"):
        # These datasets are cached via fetch_openml, which stores in sklearn's cache
        uci_cache = os.path.expanduser("~/.cache/scikit-learn/datasets")
        exists = os.path.isdir(uci_cache)
        return exists, (
            "✓ UCI-Adult will be downloaded on first use (cached locally)" 
            if exists else "(UCI-Adult will be downloaded on first use)"
        )
    
    if name_lower in ("covertype",):
        uci_cache = os.path.expanduser("~/.cache/scikit-learn/datasets")
        exists = os.path.isdir(uci_cache)
        return exists, (
            "✓ Covertype will be downloaded on first use (cached locally)"
            if exists else "(Covertype will be downloaded on first use)"
        )
    
    # Check torchvision auto-download datasets
    if name_lower in auto_download_datasets:
        path = auto_download_datasets[name_lower]
        exists = os.path.isdir(path)
        if exists:
            return True, f"✓ {dataset_name} found at {path}"
        else:
            return False, f"✗ {dataset_name} not found. Will download on first use."
    
    return False, f"Unknown dataset: {dataset_name}"


def print_dataset_status(data_dir="./data"):
    """Print download status of all supported datasets."""
    datasets = [
        "MNIST", "Fashion-MNIST", "KMNIST", 
        "CIFAR10", "CIFAR100", "SVHN",
        "Tiny-ImageNet", "UCI-Adult", "Covertype"
    ]
    
    print("\n" + "="*70)
    print("Dataset Status")
    print("="*70)
    
    for ds in datasets:
        exists, status = check_dataset_exists(ds, data_dir)
        print(f"{ds:20s} | {status}")
    
    print("="*70 + "\n")


def setup_tiny_imagenet_instructions():
    """Print detailed setup instructions for Tiny-ImageNet."""
    instructions = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   SETUP INSTRUCTIONS: Tiny-ImageNet                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Tiny-ImageNet is a 200-class subset of ImageNet with 64×64 images.
It must be manually downloaded beforehand (not auto-downloadable via torchvision).

OPTION 1: Download from official source
─────────────────────────────────────────
1. Visit: http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Download (~250 MB, extracts to ~1.4 GB)
3. Extract to: ./data/tiny-imagenet-200/
   Expected structure:
   ./data/tiny-imagenet-200/
   ├── train/
   │   ├── n00000001/
   │   ├── n00000002/
   │   └── ...
   └── val/
       ├── n00000001/
       ├── n00000002/
       └── ...

OPTION 2: Using helper script (download_tiny_imagenet.py)
──────────────────────────────────────────────────────────
   python download_tiny_imagenet.py --data-dir ./data
   
   (Script will download and extract automatically)

OPTION 3: Command line (curl + unzip)
──────────────────────────────────────
   mkdir -p ./data
   cd ./data
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   rm tiny-imagenet-200.zip

After setup, run check_dataset_exists('Tiny-ImageNet') to verify.
"""
    print(instructions)


def download_tiny_imagenet(data_dir="./data"):
    """
    Automated download and extraction of Tiny-ImageNet.
    
    Note: This is a ~250 MB download that extracts to ~1.4 GB.
    May take a few minutes depending on connection speed.
    """
    import urllib.request
    import zipfile
    
    data_dir = ensure_data_dir(data_dir)
    
    # Check if already exists
    train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
    if os.path.isdir(train_dir):
        print("✓ Tiny-ImageNet already set up at", train_dir)
        return
    
    print("\nDownloading Tiny-ImageNet from Stanford CS231N (~250 MB)...")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    
    try:
        # Download with progress
        def download_with_progress(url, filepath):
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, filepath, reporthook=_reporthook)
            print("\nDownload complete!")
        
        download_with_progress(url, zip_path)
        
        # Extract
        print(f"Extracting to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        
        # Clean up zip
        os.remove(zip_path)
        
        print("✓ Tiny-ImageNet setup complete!")
        print(f"  Location: {os.path.join(data_dir, 'tiny-imagenet-200')}")
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual setup required. Run setup_tiny_imagenet_instructions() for details.")
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _reporthook(block_num, block_size, total_size):
    """Simple download progress reporter."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, 100.0 * downloaded / total_size)
        sys.stdout.write(f'\r{percent:.1f}% ')
        sys.stdout.flush()


def prepare_datasets(data_dir="./data", download_large=False):
    """
    Check and prepare common datasets.
    
    Args:
        data_dir: Root data directory (default: ./data)
        download_large: If True, attempt to download Tiny-ImageNet (takes time!)
    """
    print("\n" + "="*70)
    print("Dataset Preparation")
    print("="*70 + "\n")
    
    # Auto-download datasets (torchvision will handle these)
    auto_datasets = ["MNIST", "Fashion-MNIST", "KMNIST", "CIFAR10", "CIFAR100", "SVHN"]
    print("Auto-downloadable datasets (torchvision):")
    for ds in auto_datasets:
        exists, status = check_dataset_exists(ds, data_dir)
        print(f"  {status}")
    
    print("\nManual-setup or large datasets:")
    
    # UCI datasets (via fetch_openml)
    print("  • UCI-Adult: Will download on first use (~5 MB)")
    print("  • Covertype: Will download on first use (~10 MB)")
    
    # Tiny-ImageNet
    exists, status = check_dataset_exists("Tiny-ImageNet", data_dir)
    print(f"  • Tiny-ImageNet:")
    if exists:
        print(f"      ✓ Found at {data_dir}/tiny-imagenet-200")
    else:
        print(f"      ✗ Not found")
        if download_large:
            print("      Attempting download...")
            download_tiny_imagenet(data_dir)
        else:
            print("      Run download_tiny_imagenet() to download (~250 MB)")
            print("      Or see setup_tiny_imagenet_instructions() for manual setup")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Simple CLI interface
    print("Dataset Utilities\n")
    print("Available functions:")
    print("  - check_dataset_exists(dataset_name, data_dir)")
    print("  - print_dataset_status(data_dir)")
    print("  - setup_tiny_imagenet_instructions()")
    print("  - download_tiny_imagenet(data_dir)")
    print("  - prepare_datasets(data_dir, download_large=False)")
