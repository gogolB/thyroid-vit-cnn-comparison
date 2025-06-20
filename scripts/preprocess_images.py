#!/usr/bin/env python3
"""
Image preprocessing pipeline for CARS thyroid images.
Applies research-recommended processing including:
- Anscombe transform
- Quality-aware preprocessing
- Percentile-based normalization
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import cv2
import tifffile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.quality_preprocessing import QualityAwarePreprocessor
from src.data.transforms import MicroscopyNormalize
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

def anscombe_transform(x: np.ndarray) -> np.ndarray:
    """
    Apply Anscombe transform to stabilize Poisson noise.
    Formula: f(x) = 2 * sqrt(x + 3/8)
    """
    return 2 * np.sqrt(x.astype(np.float32) + 3/8)

def apply_clahe(img: np.ndarray, clip_limit: float = 0.03, grid_size: tuple = (32, 32)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        img: Input image as numpy array
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
        
    Returns:
        Processed image
    """
    # Convert to 8-bit if needed
    if img.dtype != np.uint8:
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def percentile_normalization(img: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """
    Apply percentile-based normalization to an image.
    
    Args:
        img: Input image as numpy array
        low: Lower percentile
        high: Upper percentile
        
    Returns:
        Normalized image
    """
    plow = np.percentile(img, low)
    phigh = np.percentile(img, high)
    img_normalized = (img - plow) / (phigh - plow + 1e-8)
    return np.clip(img_normalized, 0, 1)

def load_image(path: Path) -> np.ndarray:
    """Load image with support for TIFF and other formats."""
    if path.suffix.lower() in ['.tif', '.tiff']:
        return tifffile.imread(str(path))
    else:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

def save_image(img: np.ndarray, path: Path):
    """Save image as PNG with necessary conversions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if img.dtype != np.uint8:
        # Convert to 8-bit: scale to 0-255 range
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
    
    cv2.imwrite(str(path), img)

def process_single_image(img_path: Path) -> np.ndarray:
    """Process a single image through the pipeline."""
    try:
        # Load image
        img = load_image(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Apply Anscombe transform
        img = anscombe_transform(img)
        
        # Apply CLAHE
        img = apply_clahe(img, clip_limit=0.03, grid_size=(32, 32))
        
        # Apply percentile normalization
        img = percentile_normalization(img, 1, 99)
        
        return img
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='CARS Thyroid Image Preprocessing Pipeline')
    parser.add_argument('--input', type=str, default='data/raw', 
                        help='Path to raw data directory')
    parser.add_argument('--output', type=str, default='data/processed', 
                        help='Path to processed data directory')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of parallel workers (not implemented)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    
    # Find all image files
    image_paths = []
    for fmt in SUPPORTED_FORMATS:
        image_paths.extend(list(input_dir.rglob(f'*{fmt}')))
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Get relative path and create output path
            rel_path = img_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.png')
            
            # Process and save image
            processed_img = process_single_image(img_path)
            save_image(processed_img, output_path)
            
        except Exception as e:
            logger.error(f"Skipping {img_path} due to error: {e}")
            continue

if __name__ == '__main__':
    main()