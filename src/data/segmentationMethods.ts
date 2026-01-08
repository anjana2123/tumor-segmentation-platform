export interface SegmentationMethod {
  id: string;
  name: string;
  category: 'classical' | 'machine-learning' | 'deep-learning';
  description: string;
  algorithm: string;
  logic: string;
  advantages: string[];
  limitations: string[];
  useCases: string[];
  computationalComplexity: string;
  accuracy: string;
  referenceYear: number;
  keyPapers: string[];
}

export const segmentationMethods: SegmentationMethod[] = [
  {
    id: 'otsu',
    name: "Otsu's Thresholding",
    category: 'classical',
    description: 'An automatic threshold selection method that maximizes inter-class variance between foreground and background pixels.',
    algorithm: 'Histogram-based thresholding using variance maximization',
    logic: 'Computes optimal threshold by maximizing between-class variance. For each threshold t, calculates class probabilities and means, then maximizes between-class variance formula.',
    advantages: [
      'Fast and computationally efficient',
      'No user parameters required',
      'Works well with bimodal intensity distributions',
      'Deterministic results'
    ],
    limitations: [
      'Struggles with multimodal distributions',
      'Sensitive to noise and artifacts',
      'Cannot handle overlapping intensity distributions',
      'No spatial information considered'
    ],
    useCases: [
      'Initial tumor boundary detection',
      'Preprocessing for complex segmentation pipelines',
      'Quick screening of contrast-enhanced tumors',
      'Educational purposes and baseline comparisons'
    ],
    computationalComplexity: 'O(n) - Linear time complexity',
    accuracy: 'Moderate (48-55% Dice score)',
    referenceYear: 1979,
    keyPapers: [
      'Otsu N. (1979) - A threshold selection method from gray-level histograms',
      'Sezgin & Sankur (2004) - Survey over image thresholding techniques'
    ]
  },
  {
    id: 'region-growing',
    name: 'Region Growing',
    category: 'classical',
    description: 'Starts from seed points and iteratively adds neighboring pixels with similar properties to form segmented regions.',
    algorithm: 'Seed-based region expansion with similarity criteria',
    logic: 'Initialize seed points in bright regions. Check neighbors and add pixels with similar intensity. Update region statistics iteratively. Continue until no more pixels can be added.',
    advantages: [
      'Preserves connectivity of segmented regions',
      'Can adapt to local intensity variations',
      'Intuitive and easy to implement',
      'Good for homogeneous tumors'
    ],
    limitations: [
      'Sensitive to seed point selection',
      'Manual intervention required for seed placement',
      'Can leak into adjacent structures with similar intensity',
      'Threshold selection affects results significantly'
    ],
    useCases: [
      'Semi-automatic tumor segmentation with user guidance',
      'Segmenting well-defined tumors with uniform intensity',
      'Interactive clinical applications',
      'Segmentation of specific tumor subregions'
    ],
    computationalComplexity: 'O(n) - Linear in number of pixels',
    accuracy: 'Good (54% Dice score)',
    referenceYear: 1994,
    keyPapers: [
      'Adams & Bischof (1994) - Seeded region growing',
      'Pohle & Toennies (2001) - Segmentation of medical images using region growing'
    ]
  },
  {
    id: 'watershed',
    name: 'Watershed Segmentation',
    category: 'classical',
    description: 'Treats grayscale image as topographic surface and finds watershed lines that separate different basins or regions.',
    algorithm: 'Morphological watershed transform',
    logic: 'Compute gradient magnitude. Find regional minima as markers. Simulate flooding from minima. Watershed lines form where water from different basins meets.',
    advantages: [
      'Can segment touching or overlapping objects',
      'Produces closed contours',
      'Works well with marker-controlled approach',
      'Captures fine details and boundaries'
    ],
    limitations: [
      'Over-segmentation in noisy images',
      'Requires preprocessing (smoothing, markers)',
      'Computationally more expensive than thresholding',
      'Sensitive to local minima in gradient image'
    ],
    useCases: [
      'Separating multiple tumor regions',
      'Detecting tumor boundaries in contrast-enhanced images',
      'Cell segmentation in microscopy',
      'Combined with other methods for refinement'
    ],
    computationalComplexity: 'O(n log n) - Using priority queue',
    accuracy: 'Good (50% Dice score)',
    referenceYear: 1991,
    keyPapers: [
      'Vincent & Soille (1991) - Watersheds in digital spaces',
      'Beucher & Meyer (1993) - Mathematical morphology in image processing'
    ]
  },
  {
    id: 'kmeans',
    name: 'K-Means Clustering',
    category: 'classical',
    description: 'Unsupervised clustering algorithm that partitions image into K clusters based on intensity similarity.',
    algorithm: 'Iterative centroid-based clustering',
    logic: 'Initialize K cluster centroids. Assign each pixel to nearest centroid. Update centroids as mean of assigned pixels. Repeat until convergence. Select cluster with highest intensity as tumor.',
    advantages: [
      'Simple and fast implementation',
      'Works well for intensity-based segmentation',
      'No need for spatial information',
      'Suitable for multi-class segmentation'
    ],
    limitations: [
      'Must specify number of clusters K in advance',
      'Sensitive to initialization',
      'Ignores spatial coherence',
      'Poor with non-spherical cluster shapes'
    ],
    useCases: [
      'Initial tissue classification (tumor, edema, healthy)',
      'Multi-region tumor segmentation',
      'Preprocessing for more sophisticated methods',
      'Unsupervised exploratory analysis'
    ],
    computationalComplexity: 'O(n * K * i) where n is pixels, K is clusters, i is iterations',
    accuracy: 'Very High (89.7% Dice score)',
    referenceYear: 1967,
    keyPapers: [
      'MacQueen (1967) - Some methods for classification and analysis',
      'Kanungo et al. (2002) - An efficient k-means clustering algorithm'
    ]
  },
  {
    id: 'morphological',
    name: 'Morphological Operations',
    category: 'classical',
    description: 'Set theory-based operations using structuring elements to extract image components useful for representation and description.',
    algorithm: 'Mathematical morphology (erosion, dilation, opening, closing)',
    logic: 'Apply threshold. Use erosion to shrink regions and remove noise. Use dilation to expand regions and fill gaps. Apply opening to remove small objects. Apply closing to fill holes.',
    advantages: [
      'Effective noise removal and gap filling',
      'Shape-based analysis capabilities',
      'Can be combined with other segmentation methods',
      'Preserves important structural features'
    ],
    limitations: [
      'Requires careful structuring element selection',
      'May distort boundaries',
      'Not a complete segmentation method alone',
      'Difficult to tune for varying tumor morphologies'
    ],
    useCases: [
      'Post-processing of segmentation results',
      'Removing false positives and small artifacts',
      'Filling holes in tumor regions',
      'Smoothing boundaries'
    ],
    computationalComplexity: 'O(n * m) where n is pixels, m is structuring element size',
    accuracy: 'Moderate (16% Dice score)',
    referenceYear: 1964,
    keyPapers: [
      'Matheron (1967) - Elements pour une theorie des milieux poreux',
      'Serra (1982) - Image Analysis and Mathematical Morphology'
    ]
  },
  {
    id: 'random-forest',
    name: 'Random Forest Classifier',
    category: 'machine-learning',
    description: 'Ensemble learning method using multiple decision trees trained on random subsets of features to classify each pixel.',
    algorithm: 'Bootstrap aggregating of decision trees with random feature selection',
    logic: 'Train 100 decision trees on bootstrap samples. Each tree uses random subset of features. For pixel classification, extract features (intensity, texture, gradient). Each tree votes. Final class is majority vote.',
    advantages: [
      'Handles high-dimensional feature spaces well',
      'Robust to overfitting through ensemble averaging',
      'Provides feature importance rankings',
      'Works with limited training data'
    ],
    limitations: [
      'Requires manual feature engineering',
      'Training data annotation needed',
      'May not capture complex spatial patterns',
      'Slower prediction than classical methods'
    ],
    useCases: [
      'Texture-based tumor classification',
      'Multi-modal MRI segmentation',
      'Regions with heterogeneous appearance',
      'Combining intensity and texture information'
    ],
    computationalComplexity: 'O(M * K * n * log(n)) training, O(M * K * log(n)) prediction',
    accuracy: 'Very High (77.6% Dice score)',
    referenceYear: 2001,
    keyPapers: [
      'Breiman (2001) - Random Forests',
      'Criminisi et al. (2012) - Decision Forests for Computer Vision'
    ]
  },
  {
    id: 'graph-cuts',
    name: 'Graph Cuts',
    category: 'machine-learning',
    description: 'Energy minimization framework that formulates segmentation as graph partitioning problem solved via max-flow/min-cut.',
    algorithm: 'Max-flow min-cut optimization on graph representation',
    logic: 'Construct graph with pixels as nodes and edges as neighborhood connections. Edge weights represent pixel similarity. Add source (foreground) and sink (background) terminals. Find minimum cut that separates source from sink.',
    advantages: [
      'Globally optimal solution within energy formulation',
      'Incorporates spatial smoothness constraints',
      'Interactive segmentation with user scribbles',
      'Well-founded mathematical framework'
    ],
    limitations: [
      'Computationally intensive for large images',
      'Limited to binary segmentation (foreground/background)',
      'Energy function design requires expertise',
      'May need user interaction for initialization'
    ],
    useCases: [
      'Interactive tumor delineation',
      'Separating tumor from surrounding tissue',
      'Refinement of initial segmentation',
      'Semi-automatic clinical workflows'
    ],
    computationalComplexity: 'O(m * n squared * C) where m is edges, n is nodes, C is max capacity',
    accuracy: 'Moderate (45.7% Dice score)',
    referenceYear: 2001,
    keyPapers: [
      'Boykov & Jolly (2001) - Interactive graph cuts',
      'Boykov et al. (2001) - Fast approximate energy minimization via graph cuts'
    ]
  },
  {
    id: 'unet',
    name: 'SAM (Segment Anything Model)',
    category: 'deep-learning',
    description: 'Vision Transformer backbone pre-trained on 11 million images with 1 billion masks. Zero-shot segmentation using automatic mask generation.',
    algorithm: 'Vision Transformer encoder with lightweight mask decoder for automatic segmentation',
    logic: 'Encode image with ViT backbone. Generate mask proposals automatically. Filter masks based on overlap with bright regions (potential tumor areas). Combine selected masks into final segmentation.',
    advantages: [
      'Pre-trained on massive dataset (11M images)',
      'Zero-shot capability without task-specific training',
      'Can segment diverse objects and patterns',
      'Robust to domain shift from natural to medical images'
    ],
    limitations: [
      'Not specifically trained on medical imaging data',
      'Computationally expensive (ViT backbone)',
      'May not understand tumor-specific patterns',
      'Requires significant GPU memory for inference'
    ],
    useCases: [
      'Exploratory medical image analysis',
      'Zero-shot tumor detection without training',
      'Baseline for transfer learning studies',
      'Quick prototyping of segmentation pipelines'
    ],
    computationalComplexity: 'O(n * p) where n is image size, p is number of prompts',
    accuracy: 'Moderate (45.8% Dice score)',
    referenceYear: 2023,
    keyPapers: [
      'Kirillov et al. (2023) - Segment Anything. arXiv preprint',
      'Dosovitskiy et al. (2021) - An Image is Worth 16x16 Words'
    ]
  },
  {
    id: 'nnunet',
    name: 'MedSAM (Medical Segment Anything)',
    category: 'deep-learning',
    description: 'SAM fine-tuned on medical images. Uses point prompts in tumor regions for targeted medical image segmentation.',
    algorithm: 'SAM with medical-specific prompting strategy using targeted point prompts',
    logic: 'Encode image with SAM. Identify likely tumor locations (bright regions above 75th percentile). Sample 10 point prompts from candidates. Generate masks conditioned on prompts. Select best mask by quality score.',
    advantages: [
      'Guided by domain knowledge for prompt placement',
      'More focused than automatic mask generation',
      'Leverages SAM pre-training',
      'Flexible prompting allows user interaction'
    ],
    limitations: [
      'Requires heuristic for prompt selection',
      'Still lacks medical-specific pre-training in this implementation',
      'Prompt quality directly affects segmentation quality',
      'Multiple prompts increase computation time'
    ],
    useCases: [
      'Point-based tumor segmentation',
      'Semi-automatic clinical workflows',
      'Interactive segmentation with clinician guidance',
      'Transfer learning baseline for medical AI'
    ],
    computationalComplexity: 'O(n + p * m) where n is encoding, p is prompts, m is mask generation',
    accuracy: 'Moderate (46.0% Dice score)',
    referenceYear: 2024,
    keyPapers: [
      'Ma et al. (2024) - Segment Anything in Medical Images',
      'Kirillov et al. (2023) - Segment Anything. arXiv preprint'
    ]
  },
  {
    id: 'attention-unet',
    name: 'SwinUNETR (Swin Transformer UNETR)',
    category: 'deep-learning',
    description: 'Hierarchical Vision Transformer with shifted window attention mechanism combined with CNN decoder for medical segmentation.',
    algorithm: 'Swin Transformer encoder with CNN decoder and skip connections',
    logic: 'Patch embedding converts image to tokens. Hierarchical Swin Transformer blocks with shifted windows extract multi-scale features. Skip connections preserve spatial details. CNN decoder upsamples and refines segmentation.',
    advantages: [
      'Efficient linear complexity through window attention',
      'Captures long-range dependencies effectively',
      'Hierarchical features at multiple scales',
      'State-of-the-art on medical imaging benchmarks when trained'
    ],
    limitations: [
      'Requires training on medical data (no pre-trained weights in this implementation)',
      'Complex architecture with many hyperparameters',
      'Memory intensive for full 3D volumes',
      'Longer inference time compared to pure CNNs'
    ],
    useCases: [
      'High-resolution medical image segmentation',
      'Multi-organ and multi-tumor segmentation',
      'Research applications with training datasets',
      'Transfer learning from pre-trained medical models'
    ],
    computationalComplexity: 'O(n) linear complexity due to window-based attention',
    accuracy: 'Moderate without training (50.6% Dice score)',
    referenceYear: 2022,
    keyPapers: [
      'Hatamizadeh et al. (2022) - Swin UNETR for Semantic Segmentation',
      'Liu et al. (2021) - Swin Transformer using Shifted Windows'
    ]
  },
  {
    id: 'deeplabv3',
    name: 'UNETR (Transformer-based UNETR)',
    category: 'deep-learning',
    description: 'Pure Vision Transformer encoder with CNN decoder. Uses self-attention to model global context throughout the image.',
    algorithm: 'Vision Transformer encoder (12 layers) with CNN decoder',
    logic: 'Tokenize image patches. Add positional embeddings. Process through 12 transformer layers with multi-head self-attention. Extract features at depths 3, 6, 9, 12. CNN decoder with skip connections upsamples to full resolution.',
    advantages: [
      'Global receptive field from the first layer',
      'Models long-range spatial dependencies',
      'Multi-scale feature extraction at different depths',
      'Proven effective when trained on medical imaging'
    ],
    limitations: [
      'Requires training on medical data (no pre-trained weights in this implementation)',
      'Computationally expensive with quadratic attention',
      'High memory requirements',
      'May overfit on small datasets'
    ],
    useCases: [
      'Research on attention mechanisms for medical imaging',
      'Large-scale medical imaging studies with training data',
      'Transfer learning from pre-trained transformers',
      'Tasks requiring global context understanding'
    ],
    computationalComplexity: 'O(n squared) quadratic in sequence length for self-attention',
    accuracy: 'Poor without training (9.4% Dice score)',
    referenceYear: 2022,
    keyPapers: [
      'Hatamizadeh et al. (2022) - UNETR Transformers for 3D Medical Segmentation',
      'Vaswani et al. (2017) - Attention is All You Need'
    ]
  },
  {
    id: 'transunet',
    name: 'SegResNet (Residual Segmentation Network)',
    category: 'deep-learning',
    description: 'ResNet-based encoder-decoder with VAE regularization. Efficient CNN architecture with residual connections.',
    algorithm: 'ResNet encoder with VAE bottleneck and upsampling decoder',
    logic: 'Initial convolution and normalization. Residual blocks with downsampling extract features. VAE bottleneck learns compressed representation. Upsampling blocks with skip connections reconstruct segmentation.',
    advantages: [
      'Efficient ResNet backbone with proven performance',
      'VAE regularization improves generalization',
      'Skip connections preserve spatial information',
      'Faster inference than transformer-based models'
    ],
    limitations: [
      'Requires training on medical data (no pre-trained weights in this implementation)',
      'Limited receptive field compared to transformers',
      'May miss long-range spatial dependencies',
      'VAE training can be unstable'
    ],
    useCases: [
      'Fast medical image segmentation after training',
      'Resource-constrained deployment scenarios',
      'Baseline CNN architecture for comparison studies',
      'Real-time clinical applications when trained'
    ],
    computationalComplexity: 'O(n) linear in number of pixels',
    accuracy: 'Very poor without training (2.4% Dice score)',
    referenceYear: 2018,
    keyPapers: [
      'Myronenko (2018) - 3D MRI brain tumor segmentation using autoencoder',
      'He et al. (2016) - Deep Residual Learning for Image Recognition'
    ]
  }
];

export const getCategoryMethods = (category: string) => {
  if (category === 'all') return segmentationMethods;
  return segmentationMethods.filter(method => method.category === category);
};

export const getMethodById = (id: string) => {
  return segmentationMethods.find(method => method.id === id);
};