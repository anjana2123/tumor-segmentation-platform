import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import pickle
import json
from datetime import datetime

from utils.brats_loader import list_brats_cases, load_brats_case

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {DEVICE}")

# Data directory
BRATS_DIR = Path("data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Results file
RESULTS_FILE = MODELS_DIR / 'training_results.json'


def load_results():
    """Load existing training results."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    """Save training results."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {RESULTS_FILE}")


def validate_case(case_path):
    """
    Check if a case has all required files.
    """
    case_path = Path(case_path)
    case_name = case_path.name
    
    # Check for FLAIR and segmentation
    flair_nii = case_path / f"{case_name}_flair.nii"
    flair_niigz = case_path / f"{case_name}_flair.nii.gz"
    seg_nii = case_path / f"{case_name}_seg.nii"
    seg_niigz = case_path / f"{case_name}_seg.nii.gz"
    
    has_flair = flair_nii.exists() or flair_niigz.exists()
    has_seg = seg_nii.exists() or seg_niigz.exists()
    
    return has_flair and has_seg


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS brain tumor segmentation.
    """
    def __init__(self, case_paths, transform=None):
        self.case_paths = case_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.case_paths)
    
    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        
        try:
            # Load image and ground truth
            image, mask, _ = load_brats_case(case_path, modality='flair')
            
            # Resize to 256x256
            image = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
            mask = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
            
            # Resize
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear'
            ).squeeze(0)
            
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), 
                size=(256, 256), 
                mode='nearest'
            ).squeeze(0)
            
            # Convert grayscale to 3-channel (repeat)
            image = image.repeat(3, 1, 1)  # (3, 256, 256)
            
            # Normalize
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {case_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)


def dice_loss(pred, target, smooth=1e-5):
    """
    Dice Loss for segmentation.
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def train_model(model_name, architecture, encoder, num_epochs=10, batch_size=4, learning_rate=1e-4, valid_paths=None):
    """
    Train a segmentation model on BraTS data.
    
    Args:
        model_name: Name for saving (e.g., 'unet_resnet34')
        architecture: 'unet', 'unetplusplus', 'deeplabv3plus', 'fpn'
        encoder: Encoder backbone (e.g., 'resnet34')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        valid_paths: Pre-validated case paths (optional)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Check if already trained
    model_path = MODELS_DIR / f"{model_name}.pth"
    if model_path.exists():
        print(f"⚠️  Model {model_name} already exists!")
        print(f"   Delete {model_path} to retrain")
        
        # Try to load and validate
        try:
            results = load_results()
            if model_name in results:
                print(f"   Previous best Dice: {results[model_name]:.4f}")
                return results[model_name]
        except:
            pass
        
        response = input(f"   Skip this model? (y/n): ").strip().lower()
        if response == 'y':
            return None
    
    # Get valid cases if not provided
    if valid_paths is None:
        all_cases = list_brats_cases(BRATS_DIR)
        case_paths = [path for _, path in all_cases]
        
        print("Validating cases...")
        valid_paths = []
        for path in tqdm(case_paths, desc="Checking cases"):
            if validate_case(path):
                valid_paths.append(path)
        
        print(f"Valid cases: {len(valid_paths)}")
    
    # Shuffle and split
    random.shuffle(valid_paths)
    split_idx = int(0.8 * len(valid_paths))
    train_paths = valid_paths[:split_idx]
    val_paths = valid_paths[split_idx:]
    
    print(f"Training cases: {len(train_paths)}")
    print(f"Validation cases: {len(val_paths)}")
    
    # Create datasets
    train_dataset = BraTSDataset(train_paths)
    val_dataset = BraTSDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    try:
        if architecture == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,  # Train from scratch for medical imaging
                in_channels=3,
                classes=1,
                activation=None
            )
        elif architecture == 'unetplusplus':
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        elif architecture == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        elif architecture == 'fpn':
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        
        model = model.to(DEVICE)
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_dice = 0.0
    
    try:
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    
                    outputs = model(images)
                    loss = dice_loss(outputs, masks)
                    
                    val_loss += loss.item()
                    
                    # Calculate Dice score
                    pred = torch.sigmoid(outputs) > 0.5
                    intersection = (pred * masks).sum()
                    dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-5)
                    val_dice += dice.item()
            
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), model_path)
                print(f"✓ Saved best model (Dice: {val_dice:.4f})")
                
                # Update results file after each improvement
                results = load_results()
                results[model_name] = float(best_val_dice)
                results[f"{model_name}_timestamp"] = datetime.now().isoformat()
                save_results(results)
        
        print(f"\n✅ Training complete! Best Dice: {best_val_dice:.4f}")
        return best_val_dice
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user!")
        if best_val_dice > 0:
            print(f"   Best model saved with Dice: {best_val_dice:.4f}")
            return best_val_dice
        return None
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if best_val_dice > 0:
            print(f"   Best model so far saved with Dice: {best_val_dice:.4f}")
            return best_val_dice
        return None


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    print("🚀 Starting training pipeline...")
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {BRATS_DIR}")
    
    # Validate cases once at the beginning
    print("\nValidating dataset...")
    all_cases = list_brats_cases(BRATS_DIR)
    case_paths = [path for _, path in all_cases]
    
    valid_paths = []
    for path in tqdm(case_paths, desc="Checking cases"):
        if validate_case(path):
            valid_paths.append(path)
    
    print(f"\n✓ Found {len(valid_paths)} valid cases out of {len(case_paths)}")
    
    # Load existing results
    results = load_results()
    print(f"\nExisting trained models: {list(results.keys())}")
    
    # Models to train
    models_to_train = [
        {
            'name': 'unet_resnet34',
            'architecture': 'unet',
            'encoder': 'resnet34',
            'epochs': 15,
            'batch_size': 4
        },
        {
            'name': 'unetplusplus_resnet34',
            'architecture': 'unetplusplus',
            'encoder': 'resnet34',
            'epochs': 15,
            'batch_size': 4
        },
        {
            'name': 'deeplabv3plus_resnet34',
            'architecture': 'deeplabv3plus',
            'encoder': 'resnet34',
            'epochs': 15,
            'batch_size': 4
        },
        {
            'name': 'fpn_resnet50',
            'architecture': 'fpn',
            'encoder': 'resnet50',
            'epochs': 15,
            'batch_size': 4
        }
    ]
    
    # Train each model with error handling
    for i, config in enumerate(models_to_train, 1):
        print(f"\n{'#'*60}")
        print(f"MODEL {i}/{len(models_to_train)}: {config['name']}")
        print(f"{'#'*60}")
        
        try:
            dice_score = train_model(
                model_name=config['name'],
                architecture=config['architecture'],
                encoder=config['encoder'],
                num_epochs=config['epochs'],
                batch_size=config['batch_size'],
                valid_paths=valid_paths  # Reuse validated paths
            )
            
            if dice_score is not None:
                results[config['name']] = float(dice_score)
                save_results(results)
                print(f"\n✅ {config['name']}: {dice_score:.4f}")
            else:
                print(f"\n⚠️  {config['name']}: Skipped or failed")
        
        except Exception as e:
            print(f"\n❌ Failed to train {config['name']}: {e}")
            print("   Continuing to next model...")
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    
    results = load_results()
    if results:
        print("\nTrained models:")
        for model_name, dice in results.items():
            if not model_name.endswith('_timestamp'):
                print(f"  {model_name}: {dice:.4f}")
    else:
        print("\nNo models successfully trained.")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Model weights in: {MODELS_DIR}")