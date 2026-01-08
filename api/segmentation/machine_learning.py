import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage import segmentation, feature, color
import pickle
from pathlib import Path

def calculate_metrics(mask):
    """Calculate dummy metrics (will be replaced with real metrics in main.py)."""
    return {
        'diceScore': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'processingTime': 0.0
    }


def extract_features(image):
    """
    Extract features from image for Random Forest.
    Same feature extraction as training.
    """
    # Convert to float32 for OpenCV
    image = image.astype(np.float32)
    
    h, w = image.shape
    features = []
    
    # 1. Original intensity
    intensity = image.flatten()
    features.append(intensity)
    
    # 2. Gradient magnitude and direction
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2).flatten()
    gradient_dir = np.arctan2(gradient_y, gradient_x).flatten()
    features.append(gradient_mag)
    features.append(gradient_dir)
    
    # 3. Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_32F).flatten()
    features.append(laplacian)
    
    # 4. Gaussian blur at multiple scales
    for sigma in [1, 2, 4]:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma).flatten()
        features.append(blurred)
    
    # 5. Local Binary Pattern
    image_uint8 = (image * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(image_uint8, P=8, R=1, method='uniform')
    features.append(lbp.flatten())
    
    # Stack all features
    features_array = np.column_stack(features)
    
    return features_array


# Load trained Random Forest model (once at import)
MODELS_DIR = Path(__file__).parent.parent / "models"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"

try:
    with open(RF_MODEL_PATH, 'rb') as f:
        RF_MODEL = pickle.load(f)
    print(f"Loaded trained Random Forest from {RF_MODEL_PATH}")
except Exception as e:
    print(f"Random Forest model not found: {e}")
    RF_MODEL = None


def random_forest_segmentation(image):
    """
    Random Forest Segmentation (TRAINED MODEL)
    
    Uses trained Random Forest classifier on extracted features.
    
    Features:
    - Intensity, gradients, texture (LBP), multi-scale blur
    
    Best for: Complex patterns learned from training data
    """
    if RF_MODEL is None:
        print("Random Forest model not trained! Run train_random_forest.py first.")
        # Return simple threshold as fallback
        threshold = np.percentile(image, 75)
        mask = (image > threshold).astype(float)
        return mask, calculate_metrics(mask)
    
    # Extract features
    features = extract_features(image)
    
    # Predict using trained model
    predictions = RF_MODEL.predict(features)
    
    # Reshape to image dimensions
    h, w = image.shape
    mask = predictions.reshape(h, w).astype(float)
    
    # Post-processing: remove small objects
    from skimage import morphology
    mask_cleaned = morphology.remove_small_objects(mask > 0.5, min_size=100)
    mask = mask_cleaned.astype(float)
    
    return mask, calculate_metrics(mask)


def graph_cuts_segmentation(image):
    """
    Graph Cuts Segmentation
    
    Algorithm:
    1. Build graph with pixels as nodes
    2. Define edge weights based on intensity similarity
    3. Use max-flow/min-cut to partition graph
    4. Implementation using GrabCut (OpenCV)
    
    Best for: Interactive segmentation with user input
    """
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Convert to 3-channel
    image_3ch = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    
    # Initialize mask for GrabCut
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    # Define rectangle around bright region (tumor)
    # Find bounding box of bright pixels
    threshold = np.percentile(image, 75)
    bright_pixels = image > threshold
    
    coords = np.column_stack(np.where(bright_pixels))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add margin
        margin = 10
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_max = min(image.shape[1], x_max + margin)
        
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Initialize models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Run GrabCut
        try:
            cv2.grabCut(image_3ch, mask, rect, bgd_model, fgd_model, 
                       iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
            
            # Extract foreground
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(float)
        except:
            # Fallback to simple threshold
            mask = bright_pixels.astype(float)
    else:
        # No bright pixels, return empty mask
        mask = np.zeros_like(image)
    
    # Post-processing
    from skimage import morphology
    mask_cleaned = morphology.remove_small_objects(mask > 0.5, min_size=100)
    mask = mask_cleaned.astype(float)
    
    return mask, calculate_metrics(mask)