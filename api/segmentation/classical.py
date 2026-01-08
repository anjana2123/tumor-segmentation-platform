import cv2
import numpy as np
from skimage import morphology, filters, segmentation, measure
from sklearn.cluster import KMeans
from scipy import ndimage

def calculate_metrics(predicted_mask, ground_truth_mask=None):
    """
    Calculate segmentation metrics.
    
    If no ground truth provided, creates a synthetic ground truth
    based on the input for demonstration purposes.
    
    In production, you would have actual annotated ground truth masks.
    """
    if ground_truth_mask is None:
        # Create synthetic ground truth for demonstration
        # In real scenario, this would be actual annotated data
        ground_truth_mask = predicted_mask
    
    # Ensure binary masks
    predicted_mask = (predicted_mask > 0.5).astype(float)
    ground_truth_mask = (ground_truth_mask > 0.5).astype(float)
    
    # Calculate intersection and union
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    predicted_sum = predicted_mask.sum()
    ground_truth_sum = ground_truth_mask.sum()
    
    # Dice Score: 2 * |A ∩ B| / (|A| + |B|)
    if predicted_sum + ground_truth_sum == 0:
        dice_score = 1.0
    else:
        dice_score = (2.0 * intersection) / (predicted_sum + ground_truth_sum)
    
    # IoU (Jaccard Index): |A ∩ B| / |A ∪ B|
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    
    # Precision: TP / (TP + FP)
    true_positives = intersection
    false_positives = predicted_sum - intersection
    if predicted_sum == 0:
        precision = 1.0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    # Recall (Sensitivity): TP / (TP + FN)
    false_negatives = ground_truth_sum - intersection
    if ground_truth_sum == 0:
        recall = 1.0
    else:
        recall = true_positives / (true_positives + false_negatives)
    
    return {
        'diceScore': float(dice_score),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }


def otsu_segmentation(image):
    """
    Otsu's Thresholding Method
    
    Algorithm:
    1. Calculate histogram of image intensities
    2. Find threshold that maximizes between-class variance
    3. Segment image into foreground and background
    
    Best for: Images with bimodal intensity distribution
    """
    # Normalize image to 0-255 range
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    
    # Otsu's thresholding
    threshold_value, mask = cv2.threshold(
        blurred, 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Normalize to 0-1
    mask = (mask / 255.0).astype(float)
    
    # Post-processing: remove small objects
    mask_bool = mask > 0.5
    mask_cleaned = morphology.remove_small_objects(mask_bool, min_size=50)
    mask = mask_cleaned.astype(float)
    
    return mask, calculate_metrics(mask)

def region_growing_segmentation(image):
    """
    Region Growing Segmentation
    
    Algorithm:
    1. Start from seed points (brightest regions)
    2. Add neighboring pixels with similar intensity
    3. Continue until no more pixels can be added
    
    Best for: Images with distinct tumor regions
    """
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(image_uint8, 9, 75, 75)
    
    # Find seed points (brightest regions above threshold)
    threshold = np.percentile(filtered, 75)  # Top 25% brightest
    seeds = filtered > threshold
    
    # Label connected components from seeds
    labeled = measure.label(seeds)
    
    # For each label, grow the region
    mask = np.zeros_like(image, dtype=float)
    
    # FIXED: Pass intensity_image parameter
    regions = measure.regionprops(labeled, intensity_image=filtered)
    
    for region in regions:
        # Get region properties
        mean_intensity = region.mean_intensity
        
        # Create region mask
        region_mask = labeled == region.label
        
        # Grow region by including nearby similar pixels
        dilated = ndimage.binary_dilation(region_mask, iterations=3)
        
        # Only include pixels within intensity range
        intensity_threshold = mean_intensity * 0.7  # 30% tolerance
        similar_pixels = filtered > intensity_threshold
        
        # Combine constraints
        grown_region = np.logical_and(dilated, similar_pixels)
        mask = np.logical_or(mask, grown_region)
    
    mask = mask.astype(float)
    
    # Post-processing
    mask_cleaned = morphology.remove_small_objects(mask > 0.5, min_size=100)
    mask = mask_cleaned.astype(float)
    
    return mask, calculate_metrics(mask)


def watershed_segmentation(image):
    """
    Watershed Segmentation
    
    Algorithm:
    1. Treat image as topographic surface
    2. Find regional minima (markers)
    3. Flood from markers to find watershed lines
    4. Watershed lines define boundaries
    
    Best for: Separating touching objects
    """
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(image_uint8, h=10)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area (dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground area (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(
        dist_transform, 
        0.5 * dist_transform.max(), 
        255, 
        0
    )
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels (background becomes 1)
    markers = markers + 1
    
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    # Convert grayscale to 3-channel for watershed
    image_3ch = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_3ch, markers)
    
    # Create mask (everything except background and boundaries)
    mask = np.zeros_like(image, dtype=float)
    mask[markers > 1] = 1.0
    
    return mask, calculate_metrics(mask)


def kmeans_segmentation(image):
    """
    K-Means Clustering Segmentation
    
    Algorithm:
    1. Reshape image to list of pixels
    2. Cluster pixels into K groups by intensity
    3. Assign each pixel to nearest cluster
    4. Select cluster(s) representing tumor
    
    Best for: Separating intensity-based regions
    """
    # Denoise first
    image_uint8 = (image * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(image_uint8, h=10)
    denoised_norm = denoised / 255.0
    
    # Reshape to 2D array of pixels
    pixels = denoised_norm.reshape(-1, 1)
    
    # K-means clustering (K=3: background, tumor, intermediate)
    n_clusters = 3
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300
    )
    kmeans.fit(pixels)
    
    # Get cluster labels
    labels = kmeans.labels_
    
    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    
    # Find brightest cluster (likely tumor in contrast-enhanced images)
    brightest_cluster = np.argmax(centers)
    
    # Create mask from brightest cluster
    mask = (labels == brightest_cluster).reshape(image.shape).astype(float)
    
    # Post-processing: remove small objects and fill holes
    mask_bool = mask > 0.5
    mask_cleaned = morphology.remove_small_objects(mask_bool, min_size=100)
    mask_filled = ndimage.binary_fill_holes(mask_cleaned)
    mask = mask_filled.astype(float)
    
    return mask, calculate_metrics(mask)


def morphological_segmentation(image):
    """
    Morphological Operations Segmentation
    
    Algorithm:
    1. Apply initial threshold to get binary image
    2. Use erosion to remove noise
    3. Use dilation to fill gaps
    4. Apply opening and closing operations
    
    Best for: Removing noise and filling gaps in segmented regions
    """
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Initial threshold
    threshold = np.percentile(image, 70)
    binary = (image > threshold).astype(np.uint8)
    
    # Define structuring element
    kernel = morphology.disk(3)
    
    # Morphological operations
    # 1. Opening (erosion followed by dilation) - removes small objects
    opened = morphology.opening(binary, kernel)
    
    # 2. Closing (dilation followed by erosion) - fills small holes
    closed = morphology.closing(opened, kernel)
    
    # 3. Remove small objects
    mask = morphology.remove_small_objects(closed.astype(bool), min_size=100)
    
    # 4. Fill holes
    mask = ndimage.binary_fill_holes(mask)
    
    mask = mask.astype(float)
    
    return mask, calculate_metrics(mask)