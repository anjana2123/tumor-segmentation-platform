import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from tqdm import tqdm
from skimage import feature
import cv2

from utils.brats_loader import list_brats_cases, load_brats_case

# Paths
BRATS_DIR = Path("data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("🌲 Training Random Forest Classifier on BraTS Dataset")
print("="*60)


def extract_features(image):
    """
    Extract features from image for Random Forest.
    
    Features:
    - Intensity
    - Gradient magnitude
    - Gradient direction
    - Laplacian
    - Local binary patterns (texture)
    - Gaussian blur (different scales)
    """
    # CRITICAL FIX: Convert to float32 for OpenCV
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
    
    # 3. Laplacian (edge detection)
    laplacian = cv2.Laplacian(image, cv2.CV_32F).flatten()
    features.append(laplacian)
    
    # 4. Gaussian blur at multiple scales
    for sigma in [1, 2, 4]:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma).flatten()
        features.append(blurred)
    
    # 5. Local Binary Pattern (texture)
    image_uint8 = (image * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(image_uint8, P=8, R=1, method='uniform')
    features.append(lbp.flatten())
    
    # Stack all features
    features_array = np.column_stack(features)
    
    return features_array


print("Loading and validating cases...")
all_cases = list_brats_cases(BRATS_DIR)

# Use subset for training (Random Forest trains on fewer samples)
num_training_cases = 100  # Use 100 cases for speed
print(f"Using {num_training_cases} cases for training")

training_cases = all_cases[:num_training_cases]

# Extract features and labels
print("Extracting features from images...")
X_list = []
y_list = []

successful_cases = 0
failed_cases = 0

for case_id, case_path in tqdm(training_cases, desc="Processing cases"):
    try:
        # Load image and ground truth
        image, mask, _ = load_brats_case(case_path, modality='flair')
        
        # Extract features
        features = extract_features(image)
        labels = mask.flatten()
        
        X_list.append(features)
        y_list.append(labels)
        
        successful_cases += 1
        
    except Exception as e:
        failed_cases += 1
        if failed_cases <= 3:  # Only print first 3 errors
            print(f"\nError with {case_id}: {e}")
        continue

print(f"\nSuccessful: {successful_cases}, Failed: {failed_cases}")

if len(X_list) == 0:
    print("❌ No cases processed successfully! Cannot train.")
    exit(1)

# Combine all data
print("Combining data...")
X = np.vstack(X_list)
y = np.concatenate(y_list)

print(f"Total samples: {X.shape[0]:,}")
print(f"Features per sample: {X.shape[1]}")
print(f"Positive samples: {y.sum():,} ({y.mean()*100:.2f}%)")

# Balance dataset (too many background pixels)
print("Balancing dataset...")
positive_idx = np.where(y == 1)[0]
negative_idx = np.where(y == 0)[0]

# Sample equal number of positive and negative
num_samples = min(len(positive_idx), len(negative_idx), 500000)  # Limit to 500k each
print(f"Using {num_samples:,} samples per class")

np.random.seed(42)
np.random.shuffle(positive_idx)
np.random.shuffle(negative_idx)

balanced_idx = np.concatenate([
    positive_idx[:num_samples],
    negative_idx[:num_samples]
])

np.random.shuffle(balanced_idx)

X_balanced = X[balanced_idx]
y_balanced = y[balanced_idx]

print(f"Balanced dataset: {X_balanced.shape[0]:,} samples")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Testing samples: {X_test.shape[0]:,}")

# Train Random Forest
print("\nTraining Random Forest...")
print("Parameters:")
print("  - n_estimators: 100")
print("  - max_depth: 20")
print("  - min_samples_split: 100")
print("  - n_jobs: -1 (all cores)")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=100,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf.fit(X_train, y_train)

# Evaluate
print("\nEvaluating...")
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Feature importance
feature_names = [
    'Intensity', 'Gradient_Mag', 'Gradient_Dir', 'Laplacian',
    'Gaussian_σ1', 'Gaussian_σ2', 'Gaussian_σ4', 'LBP'
]
importances = rf.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# Save model
model_path = MODELS_DIR / 'random_forest.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"\n✅ Model saved to: {model_path}")
print("="*60)
print("Random Forest training complete!")