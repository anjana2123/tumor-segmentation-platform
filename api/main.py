from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import base64
import time
from pathlib import Path

# Auto-download models on startup
import subprocess
import sys

MODELS_DIR = Path(__file__).parent / "models"
SAM_MODEL = MODELS_DIR / "sam" / "sam_vit_b.pth"

if not SAM_MODEL.exists():
    print("MODELS NOT FOUND - DOWNLOADING FROM GOOGLE DRIVE")
    print("This is a one-time process (2-3 minutes)...")
    try:
        subprocess.run([sys.executable, "download_models.py"], cwd=Path(__file__).parent, check=True)
        print("=" * 60)
        print("MODELS READY - STARTING API SERVER")
        print("=" * 60)
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Continuing without all models...")

# Import segmentation methods
from segmentation.classical import (
    otsu_segmentation,
    region_growing_segmentation,
    watershed_segmentation,
    kmeans_segmentation,
    morphological_segmentation
)
from segmentation.machine_learning import (
    random_forest_segmentation,
    graph_cuts_segmentation
)
from segmentation.deep_learning import (
    unet_segmentation,
    nnunet_segmentation,
    attention_unet_segmentation,
    deeplabv3_segmentation,
    transunet_segmentation
)

# Import BraTS loader
from utils.brats_loader import list_brats_cases, load_brats_case, get_case_info

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Method registry
METHOD_REGISTRY = {
    'otsu': otsu_segmentation,
    'region-growing': region_growing_segmentation,
    'watershed': watershed_segmentation,
    'kmeans': kmeans_segmentation,
    'morphological': morphological_segmentation,
    'random-forest': random_forest_segmentation,
    'graph-cuts': graph_cuts_segmentation,
    'unet': unet_segmentation,
    'nnunet': nnunet_segmentation,
    'attention-unet': attention_unet_segmentation,
    'deeplabv3': deeplabv3_segmentation,
    'transunet': transunet_segmentation,
}

# BraTS data directory
'''BRATS_DATA_DIR = Path(__file__).parent / "data" / "brats" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"'''

FULL_DATASET = Path(__file__).parent / "data" / "brats" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
DEPLOY_DATASET = Path(__file__).parent / "data" / "brats_deploy" / "MICCAI_BraTS2020_TrainingData"

# Use full dataset if available (local), otherwise use deployment dataset (Railway)
if FULL_DATASET.exists():
    BRATS_DATA_DIR = FULL_DATASET
    print(f"Using FULL dataset: {len(list(FULL_DATASET.iterdir()))} cases")
else:
    BRATS_DATA_DIR = DEPLOY_DATASET
    print(f"Using DEPLOYMENT dataset: 20 cases")

def calculate_real_metrics(predicted_mask, ground_truth_mask):
    """
    Calculate REAL metrics using ground truth.
    
    Args:
        predicted_mask: Algorithm's prediction (0-1)
        ground_truth_mask: Real annotation from radiologist (0-1)
    
    Returns:
        dict with Dice, IoU, Precision, Recall
    """
    # Binarize masks
    pred = (predicted_mask > 0.5).astype(float)
    gt = (ground_truth_mask > 0.5).astype(float)
    
    # Calculate metrics
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    # Dice Score: 2 * |A ∩ B| / (|A| + |B|)
    if pred_sum + gt_sum == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = (2.0 * intersection) / (pred_sum + gt_sum)
    
    # IoU: |A ∩ B| / |A ∪ B|
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
    
    # Precision: TP / (TP + FP)
    if pred_sum == 0:
        precision = 0.0
    else:
        precision = intersection / pred_sum
    
    # Recall: TP / (TP + FN)
    if gt_sum == 0:
        recall = 0.0
    else:
        recall = intersection / gt_sum
    
    return {
        'diceScore': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }

def mask_to_base64(mask):
    """Convert mask to base64 image."""
    # Convert to 0-255 range
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Convert to RGB for better visualization
    mask_rgb = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_HOT)
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', mask_rgb)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{mask_base64}"

@app.get("/")
def read_root():
    return {"message": "Tumor Segmentation API with BraTS Dataset!"}

@app.get("/cases")
def get_cases():
    """List all available BraTS cases."""
    try:
        cases = list_brats_cases(BRATS_DATA_DIR)
        
        # Get info for first few cases (for display)
        cases_info = []
        for case_id, case_path in cases[:50]:  # Limit to first 50 for speed
            try:
                info = get_case_info(case_path)
                cases_info.append({
                    'id': case_id,
                    'path': case_path,
                    'tumorPercentage': info['tumor_percentage'],
                    'numSlices': info['num_slices'],
                    'bestSlice': info['best_slice']
                })
            except Exception as e:
                print(f"Error getting info for {case_id}: {e}")
                continue
        
        return {"cases": cases_info, "total": len(cases)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preview-case")
def preview_case(
    case_id: str = Form(...),
    slice_idx: int = Form(None)
):
    """
    Preview a BraTS case without running segmentation.
    
    Args:
        case_id: BraTS case ID
        slice_idx: Optional specific slice
    """
    try:
        # Find case path
        case_path = BRATS_DATA_DIR / case_id
        if not case_path.exists():
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
        
        # Load image and ground truth
        image, ground_truth, used_slice_idx = load_brats_case(
            case_path, 
            modality='flair',
            slice_idx=slice_idx
        )
        
        # Convert image to base64
        image_uint8 = (image * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', image_uint8)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        original_image = f"data:image/png;base64,{image_base64}"
        
        # Convert ground truth to base64
        gt_base64 = mask_to_base64(ground_truth)
        
        return {
            'originalImage': original_image,
            'groundTruth': gt_base64,
            'caseId': case_id,
            'sliceIdx': int(used_slice_idx)
        }
    
    except Exception as e:
        print(f"Preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment-brats")
def segment_brats_case(
    case_id: str = Form(...),
    methods: str = Form(...),
    slice_idx: int = Form(None)
):
    """
    Segment a BraTS case with selected methods.
    
    Args:
        case_id: BraTS case ID (e.g., 'BraTS20_Training_001')
        methods: Comma-separated method IDs
        slice_idx: Optional specific slice, otherwise uses best slice
    """
    try:
        # Find case path
        case_path = BRATS_DATA_DIR / case_id
        if not case_path.exists():
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
        
        # Load image and ground truth
        image, ground_truth, used_slice_idx = load_brats_case(
            case_path, 
            modality='flair',  # FLAIR is best for tumor visibility
            slice_idx=slice_idx
        )
        
        # Convert image to base64 for frontend
        image_uint8 = (image * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', image_uint8)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        original_image = f"data:image/png;base64,{image_base64}"
        
        # Convert ground truth to base64
        gt_base64 = mask_to_base64(ground_truth)
        
        # Parse methods
        method_ids = [m.strip() for m in methods.split(',')]
        
        # Run segmentation for each method
        results = []
        for method_id in method_ids:
            if method_id not in METHOD_REGISTRY:
                continue
            
            method_func = METHOD_REGISTRY[method_id]
            
            # Time the segmentation
            start_time = time.time()
            
            try:
                # Run segmentation
                mask, _ = method_func(image)  # Ignore old metrics
                
                # Calculate REAL metrics using ground truth
                metrics = calculate_real_metrics(mask, ground_truth)
                metrics['processingTime'] = time.time() - start_time
                
                # Convert mask to base64
                mask_base64 = mask_to_base64(mask)
                
                results.append({
                    'methodId': method_id,
                    'segmentedImage': mask_base64,
                    'metrics': metrics
                })
                
                print(f"{method_id}: Dice={metrics['diceScore']:.3f}, Time={metrics['processingTime']:.2f}s")
            
            except Exception as e:
                print(f"Error in {method_id}: {e}")
                continue
        
        return {
            'results': results,
            'originalImage': original_image,
            'groundTruth': gt_base64,
            'caseId': case_id,
            'sliceIdx': int(used_slice_idx)
        }
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)