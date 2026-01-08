import nibabel as nib
import numpy as np
import os
from pathlib import Path
import cv2

def normalize_image(image):
    """Normalize image to 0-1 range."""
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image

def load_brats_case(case_path, modality='flair', slice_idx=None):
    """
    Load a single BraTS case.
    
    Args:
        case_path: Path to case directory (e.g., 'BraTS20_Training_001')
        modality: Which MRI sequence ('flair', 't1', 't1ce', 't2')
        slice_idx: Which slice to extract, if None uses middle slice with most tumor
    
    Returns:
        image: 2D normalized image (H, W) in range 0-1
        ground_truth: 2D binary mask (H, W) with tumor annotation
        slice_idx: The slice index used
    """
    case_path = Path(case_path)
    case_name = case_path.name
    
    # Try both .nii and .nii.gz extensions
    modality_path = case_path / f"{case_name}_{modality}.nii"
    if not modality_path.exists():
        modality_path = case_path / f"{case_name}_{modality}.nii.gz"
    
    if not modality_path.exists():
        raise FileNotFoundError(f"Modality file not found: {modality_path}")
    
    # Load 3D volume
    img_nib = nib.load(str(modality_path))
    image_3d = img_nib.get_fdata()
    
    # Load ground truth segmentation
    seg_path = case_path / f"{case_name}_seg.nii"
    if not seg_path.exists():
        seg_path = case_path / f"{case_name}_seg.nii.gz"
    
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
    
    seg_nib = nib.load(str(seg_path))
    seg_3d = seg_nib.get_fdata()
    
    # Find slice with most tumor if slice_idx not specified
    if slice_idx is None:
        # Sum tumor pixels in each slice
        tumor_per_slice = np.sum(seg_3d > 0, axis=(0, 1))
        # Get slice with most tumor
        slice_idx = np.argmax(tumor_per_slice)
        
        # If no tumor found, use middle slice
        if tumor_per_slice[slice_idx] == 0:
            slice_idx = image_3d.shape[2] // 2
    
    # Extract 2D slice
    image_2d = image_3d[:, :, slice_idx]
    seg_2d = seg_3d[:, :, slice_idx]
    
    # Normalize image to 0-1
    image_2d = normalize_image(image_2d)
    
    # Convert segmentation to binary (any tumor class = 1)
    # BraTS has multiple tumor classes (1=necrotic, 2=edema, 4=enhancing)
    # We combine all into single tumor mask
    ground_truth = (seg_2d > 0).astype(np.float32)
    
    return image_2d, ground_truth, slice_idx

def list_brats_cases(data_dir):
    """
    List all available BraTS cases.
    
    Args:
        data_dir: Path to BraTS data directory
    
    Returns:
        List of (case_id, case_path) tuples
    """
    data_dir = Path(data_dir)
    
    # Find all case directories
    cases = []
    for case_dir in sorted(data_dir.iterdir()):
        if case_dir.is_dir() and case_dir.name.startswith('BraTS'):
            case_id = case_dir.name
            cases.append((case_id, str(case_dir)))
    
    return cases

def get_case_info(case_path):
    """
    Get information about a BraTS case.
    
    Returns:
        dict with case statistics
    """
    case_path = Path(case_path)
    case_name = case_path.name
    
    # Load segmentation to get tumor info
    seg_path = case_path / f"{case_name}_seg.nii"
    if not seg_path.exists():
        seg_path = case_path / f"{case_name}_seg.nii.gz"
    
    seg_nib = nib.load(str(seg_path))
    seg_3d = seg_nib.get_fdata()
    
    # Calculate statistics
    total_voxels = seg_3d.size
    tumor_voxels = np.sum(seg_3d > 0)
    tumor_percentage = (tumor_voxels / total_voxels) * 100
    
    # Find slice with most tumor
    tumor_per_slice = np.sum(seg_3d > 0, axis=(0, 1))
    best_slice = np.argmax(tumor_per_slice)
    
    return {
        'case_name': case_name,
        'volume_shape': seg_3d.shape,
        'tumor_voxels': int(tumor_voxels),
        'tumor_percentage': float(tumor_percentage),
        'best_slice': int(best_slice),
        'num_slices': seg_3d.shape[2]
    }