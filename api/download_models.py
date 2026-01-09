import gdown
from pathlib import Path
import time
import tarfile

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
SAM_DIR = MODELS_DIR / "sam"
SAM_DIR.mkdir(exist_ok=True)

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using gdown."""
    if destination.exists():
        size_mb = destination.stat().st_size / 1024 / 1024
        if size_mb > 0.1:  # File exists and is not empty
            print(f"✓ {destination.name} already exists ({size_mb:.1f} MB)")
            return True
        else:
            destination.unlink()  # Delete empty file
    
    print(f"Downloading {destination.name}...")
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        gdown.download(url, str(destination), quiet=False)
        
        size_mb = destination.stat().st_size / 1024 / 1024
        if size_mb < 0.1:
            print(f"Download failed for {destination.name}")
            destination.unlink()
            return False
        
        print(f"Downloaded {destination.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"Error downloading {destination.name}: {e}")
        return False

# Google Drive File IDs
MODELS = {
    'sam_vit_b.pth': {
        'file_id': '148nM1-VXEAIEoB86j9G3lk1BH0Rd7Ebz',
        'destination': SAM_DIR / 'sam_vit_b.pth'
    },
    'random_forest.pkl': {
        'file_id': '12sn-af5zCowegHLdEmVNvPS8tDqkhVtH',
        'destination': MODELS_DIR / 'random_forest.pkl'
    },
    'unet_resnet34.pth': {
        'file_id': '18f5DJqAT1uZnUHCkGEGsPaVxHGP9Pvgr',
        'destination': MODELS_DIR / 'unet_resnet34.pth'
    },
    'unetplusplus_resnet34.pth': {
        'file_id': '1uKngKhj9rAHFYAM303MgHCMIkvkRHi3C',
        'destination': MODELS_DIR / 'unetplusplus_resnet34.pth'
    },
    'deeplabv3plus_resnet34.pth': {
        'file_id': '1Xtj7dgT_JO_h0k0WCN19YAwABt8z0m_D',
        'destination': MODELS_DIR / 'deeplabv3plus_resnet34.pth'
    },
    'fpn_resnet50.pth': {
        'file_id': '1EceMjdDLmW6Dwa1ZtQxYr-c3WL96AVMS',
        'destination': MODELS_DIR / 'fpn_resnet50.pth'
    },
    'training_results.json': {
        'file_id': '13cfP5M1Gaskz4TnG8x37SoiZOmOuhXe7',
        'destination': MODELS_DIR / 'training_results.json'
    }
}

# Dataset download
DATASET_TAR = Path(__file__).parent / "brats_20_cases.tar.gz"
DATASET_DIR = Path(__file__).parent / "data" / "brats_deploy"

def download_and_extract_dataset():
    """Download and extract BraTS dataset."""
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"✓ Dataset already exists")
        return True
    
    # Download tar.gz
    if not DATASET_TAR.exists():
        print("Downloading BraTS dataset (165 MB)...")
        try:
            url = "https://drive.google.com/uc?id=1Cf8IMYoe9Ps0zHU2_HhnIu_IlVt07HZ7"
            gdown.download(url, str(DATASET_TAR), quiet=False)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    # Extract
    print(f"Extracting dataset to {DATASET_DIR}...")
    try:
        with tarfile.open(DATASET_TAR, 'r:gz') as tar:
            tar.extractall(Path(__file__).parent)
        
        # Remove tar file to save space
        DATASET_TAR.unlink()
        print("Dataset ready!")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

if __name__ == "__main__":
    print("DOWNLOADING MODELS AND DATASET FROM GOOGLE DRIVE")
    
    start_time = time.time()
    
    # Download models
    for name, info in MODELS.items():
        try:
            download_from_gdrive(info['file_id'], info['destination'])
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    # Download and extract dataset
    try:
        download_and_extract_dataset()
    except Exception as e:
        print(f"Error with dataset: {e}")
    
    elapsed = time.time() - start_time
    print(f"Complete! Total time: {elapsed:.1f} seconds")