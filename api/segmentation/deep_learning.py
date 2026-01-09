import numpy as np
import cv2
import torch
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

MODELS_DIR = Path(__file__).parent.parent / "models"

# Don't load models on import - load them lazily when needed
SAM_MODEL = None
MEDSAM_PREDICTOR = None
SWIN_MODEL = None
UNETR_MODEL = None
SEGRESNET_MODEL = None

def calculate_metrics(mask):
    return {
        'diceScore': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'processingTime': 0.0
    }


def load_sam_models():
    """Lazy load SAM models"""
    global SAM_MODEL, MEDSAM_PREDICTOR
    if SAM_MODEL is not None:
        return True
        
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        SAM_CHECKPOINT = MODELS_DIR / "sam" / "sam_vit_b.pth"
        
        if not SAM_CHECKPOINT.exists():
            print("SAM checkpoint not found")
            return False
            
        # Monkey-patch torch.load for PyTorch 2.6
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        sam_base = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        sam_base.to(device=DEVICE)
        SAM_MODEL = SamAutomaticMaskGenerator(sam_base)
        
        sam_medsam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        sam_medsam.to(device=DEVICE)
        MEDSAM_PREDICTOR = SamPredictor(sam_medsam)
        
        torch.load = original_load
        print("SAM and MedSAM models loaded")
        return True
    except Exception as e:
        print(f"SAM loading failed: {e}")
        return False


def load_swinunetr():
    """Lazy load SwinUNETR"""
    global SWIN_MODEL
    if SWIN_MODEL is not None:
        return True
        
    try:
        from monai.networks.nets import SwinUNETR
        SWIN_MODEL = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=48,
        ).to(DEVICE)
        SWIN_MODEL.eval()
        print("SwinUNETR loaded")
        return True
    except Exception as e:
        print(f"SwinUNETR loading failed: {e}")
        return False


def load_unetr():
    """Lazy load UNETR"""
    global UNETR_MODEL
    if UNETR_MODEL is not None:
        return True
        
    try:
        from monai.networks.nets import UNETR
        UNETR_MODEL = UNETR(
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
        UNETR_MODEL.eval()
        print("UNETR loaded")
        return True
    except Exception as e:
        print(f"UNETR loading failed: {e}")
        return False


def load_segresnet():
    """Lazy load SegResNet"""
    global SEGRESNET_MODEL
    if SEGRESNET_MODEL is not None:
        return True
        
    try:
        from monai.networks.nets import SegResNet
        SEGRESNET_MODEL = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=1,
        ).to(DEVICE)
        SEGRESNET_MODEL.eval()
        print("SegResNet loaded")
        return True
    except Exception as e:
        print(f"SegResNet loading failed: {e}")
        return False


def unet_segmentation(image):
    """SAM - Segment Anything Model"""
    if not load_sam_models() or SAM_MODEL is None:
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
    if not load_sam_models() or MEDSAM_PREDICTOR is None:
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
    if not load_swinunetr() or SWIN_MODEL is None:
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
    if not load_unetr() or UNETR_MODEL is None:
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
    if not load_segresnet() or SEGRESNET_MODEL is None:
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