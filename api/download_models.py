import os
import requests
from pathlib import Path
import time

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
SAM_DIR = MODELS_DIR / "sam"
SAM_DIR.mkdir(exist_ok=True)

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive."""
    if destination.exists():
        print(f"{destination.name} already exists ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    
    print(f"Downloading {destination.name}")
    
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
    
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
    size_mb = destination.stat().st_size / 1024 / 1024
    print(f"Downloaded {destination.name} ({size_mb:.1f} MB)")
    return True

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
        print(f"Dataset already exists ({len(list(DATASET_DIR.rglob('*')))} files)")
        return True
    
    # Download tar.gz
    if not DATASET_TAR.exists():
        print("Downloading BraTS dataset sample (165 MB)...")
        download_from_gdrive('1Cf8IMYoe9Ps0zHU2_HhnIu_IlVt07HZ7', DATASET_TAR)
    
    # Extract
    print(f"Extracting dataset to {DATASET_DIR}...")
    import tarfile
    with tarfile.open(DATASET_TAR, 'r:gz') as tar:
        tar.extractall(Path(__file__).parent)
    
    # Remove tar file to save space
    DATASET_TAR.unlink()
    print("Dataset ready!")
    return True

'''if __name__ == "__main__":
    print("DOWNLOADING MODELS FROM GOOGLE DRIVE")
    
    start_time = time.time()
    
    for name, info in MODELS.items():
        try:
            download_from_gdrive(info['file_id'], info['destination'])
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    elapsed = time.time() - start_time
    print(f"Complete! Total time: {elapsed:.1f} seconds")'''

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
