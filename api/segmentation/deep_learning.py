import numpy as np
import cv2
import torch
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

MODELS_DIR = Path(__file__).parent.parent / "models"

def calculate_metrics(mask):
    return {
        'diceScore': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'processingTime': 0.0
    }


# SAM Model
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_CHECKPOINT = MODELS_DIR / "sam" / "sam_vit_b.pth"
    
    if SAM_CHECKPOINT.exists():
        sam_base = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        sam_base.to(device=DEVICE)
        SAM_MODEL = SamAutomaticMaskGenerator(sam_base)
        
        sam_medsam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        sam_medsam.to(device=DEVICE)
        MEDSAM_PREDICTOR = SamPredictor(sam_medsam)
        
        SAM_AVAILABLE = True
        print("SAM and MedSAM models loaded")
    else:
        SAM_AVAILABLE = False
        SAM_MODEL = None
        MEDSAM_PREDICTOR = None
        print("SAM checkpoint not found")
except Exception as e:
    SAM_AVAILABLE = False
    SAM_MODEL = None
    MEDSAM_PREDICTOR = None
    print(f"SAM not available: {e}")


# MONAI Models
try:
    from monai.networks.nets import SwinUNETR
    
    try:
        # Try newer MONAI API
        swin_model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=48,
        ).to(DEVICE)
    except TypeError:
        # Try older MONAI API
        swin_model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=3,
        ).to(DEVICE)
    
    swin_model.eval()
    SWIN_MODEL = swin_model
    SWINUNETR_AVAILABLE = True
    print("SwinUNETR loaded")
except Exception as e:
    SWINUNETR_AVAILABLE = False
    SWIN_MODEL = None
    print(f"SwinUNETR not available: {e}")
    
'''
try:
    from monai.networks.nets import SwinUNETR
    
    swin_model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,
    ).to(DEVICE)
    swin_model.eval()
    SWIN_MODEL = swin_model
    SWINUNETR_AVAILABLE = True
    print("SwinUNETR loaded")
except Exception as e:
    SWINUNETR_AVAILABLE = False
    SWIN_MODEL = None
    print(f"SwinUNETR not available: {e}")'''


try:
    from monai.networks.nets import UNETR
    
    unetr_model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
    ).to(DEVICE)
    unetr_model.eval()
    UNETR_MODEL = unetr_model
    UNETR_AVAILABLE = True
    print("UNETR loaded")
except Exception as e:
    UNETR_AVAILABLE = False
    UNETR_MODEL = None
    print(f"UNETR not available: {e}")


try:
    from monai.networks.nets import SegResNet
    
    segresnet_model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)
    segresnet_model.eval()
    SEGRESNET_MODEL = segresnet_model
    SEGRESNET_AVAILABLE = True
    print("SegResNet loaded")
except Exception as e:
    SEGRESNET_AVAILABLE = False
    SEGRESNET_MODEL = None
    print(f"SegResNet not available: {e}")


def unet_segmentation(image):
    """SAM - Segment Anything Model"""
    if not SAM_AVAILABLE or SAM_MODEL is None:
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
    
    try:
        image_uint8 = (image * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        
        masks = SAM_MODEL.generate(image_rgb)
        
        if len(masks) == 0:
            return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
        
        final_mask = np.zeros_like(image)
        threshold = np.percentile(image, 60)
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            overlap = np.logical_and(mask, image > threshold)
            if mask.sum() > 0 and overlap.sum() / mask.sum() > 0.3:
                final_mask = np.logical_or(final_mask, mask)
        
        return final_mask.astype(float), calculate_metrics(final_mask.astype(float))
    
    except Exception as e:
        print(f"SAM error: {e}")
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))


def nnunet_segmentation(image):
    """MedSAM - Medical Segment Anything"""
    if not SAM_AVAILABLE or MEDSAM_PREDICTOR is None:
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
    
    try:
        image_uint8 = (image * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        
        MEDSAM_PREDICTOR.set_image(image_rgb)
        
        threshold = np.percentile(image, 75)
        bright_pixels = np.where(image > threshold)
        
        if len(bright_pixels[0]) == 0:
            return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
        
        num_points = min(10, len(bright_pixels[0]))
        indices = np.random.choice(len(bright_pixels[0]), num_points, replace=False)
        points = np.column_stack([bright_pixels[1][indices], bright_pixels[0][indices]])
        labels = np.ones(num_points)
        
        masks, scores, _ = MEDSAM_PREDICTOR.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx].astype(float)
        
        return final_mask, calculate_metrics(final_mask)
    
    except Exception as e:
        print(f"MedSAM error: {e}")
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))


def attention_unet_segmentation(image):
    """SwinUNETR - Swin Transformer U-Net"""
    if not SWINUNETR_AVAILABLE or SWIN_MODEL is None:
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
    
    try:
        h, w = image.shape
        image_resized = cv2.resize(image, (96, 96))
        
        volume = np.stack([image_resized] * 96, axis=0)
        volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume = volume.to(DEVICE)
        
        with torch.no_grad():
            output = SWIN_MODEL(volume)
            output = torch.sigmoid(output)
        
        mask = output[0, 0, 48, :, :].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.5).astype(float)
        
        return mask, calculate_metrics(mask)
    
    except Exception as e:
        print(f"SwinUNETR error: {e}")
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))


def deeplabv3_segmentation(image):
    """UNETR - Transformer-based U-Net"""
    if not UNETR_AVAILABLE or UNETR_MODEL is None:
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
    
    try:
        h, w = image.shape
        image_resized = cv2.resize(image, (96, 96))
        
        volume = np.stack([image_resized] * 96, axis=0)
        volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume = volume.to(DEVICE)
        
        with torch.no_grad():
            output = UNETR_MODEL(volume)
            output = torch.sigmoid(output)
        
        mask = output[0, 0, 48, :, :].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.5).astype(float)
        
        return mask, calculate_metrics(mask)
    
    except Exception as e:
        print(f"UNETR error: {e}")
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))


def transunet_segmentation(image):
    """SegResNet - Residual Segmentation Network"""
    if not SEGRESNET_AVAILABLE or SEGRESNET_MODEL is None:
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))
    
    try:
        h, w = image.shape
        image_resized = cv2.resize(image, (96, 96))
        
        volume = np.stack([image_resized] * 96, axis=0)
        volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume = volume.to(DEVICE)
        
        with torch.no_grad():
            output = SEGRESNET_MODEL(volume)
            output = torch.sigmoid(output)
        
        mask = output[0, 0, 48, :, :].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.5).astype(float)
        
        return mask, calculate_metrics(mask)
    
    except Exception as e:
        print(f"SegResNet error: {e}")
        return np.zeros_like(image), calculate_metrics(np.zeros_like(image))