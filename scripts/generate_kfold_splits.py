import json
import re
from pathlib import Path
from collections import defaultdict


def extract_patient_id(path: Path) -> str:
    """
    Extract patient ID from filename using regex patterns.
    
    Args:
        path: Path object for the image file
        
    Returns:
        Extracted patient ID string (e.g., 'P1')
        
    Raises:
        ValueError: If filename doesn't match expected patterns
    """
    name = path.name
    # Match cancerous image pattern: Cancer_P(\d+)_F\d+_I\d+\.png
    cancer_match = re.match(r'Cancer_P(\d+)_F\d+_I\d+\.png', name)
    if cancer_match:
        return f"Cancer_P{cancer_match.group(1)}"

    # Match normal image pattern: Normal_P(\d+)_F\d+_I\d+\.png
    normal_match = re.match(r'Normal_P(\d+)_F\d+_I\d+\.png', name)
    if normal_match:
        return f"Normal_P{normal_match.group(1)}"
    
    raise ValueError(f"Filename {name} doesn't match expected patterns")


def main():
    """
    Generate 10-fold leave-one-patient-out splits
    from thyroid ultrasound images.
    Saves each split as JSON in data/splits directory.
    """
    # Define directories
    cancer_dir = Path("data/processed/cancerous")
    normal_dir = Path("data/processed/normal")
    output_dir = Path("data/splits")
    
    # Create output directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate directories exist
    if not cancer_dir.is_dir():
        raise FileNotFoundError(f"Cancer directory not found: {cancer_dir}")
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal directory not found: {normal_dir}")
    
    # Group images by patient ID
    patient_images = defaultdict(list)
    
    # Process cancerous images
    for img_path in cancer_dir.glob("*.png"):
        if not img_path.is_file():
            continue
        try:
            pid = extract_patient_id(img_path)
            patient_images[pid].append(str(img_path))
        except ValueError as e:
            print(f"Skipping {img_path}: {str(e)}")
    
    # Process normal images
    for img_path in normal_dir.glob("*.png"):
        if not img_path.is_file():
            continue
        try:
            pid = extract_patient_id(img_path)
            patient_images[pid].append(str(img_path))
        except ValueError as e:
            print(f"Skipping {img_path}: {str(e)}")
    
    # Check we have at least 10 patients
    patient_ids = sorted(patient_images.keys())
    if len(patient_ids) < 10:
        raise ValueError(
            f"Need at least 10 patients, found {len(patient_ids)}"
        )
    
    # Generate 10 folds (leave-one-patient-out)
    # Only use first 10 patients
    for i, test_patient in enumerate(patient_ids[:10], 1):
        train = []
        test = []
        
        for pid, paths in patient_images.items():
            if pid == test_patient:
                test.extend(paths)
            else:
                train.extend(paths)
        
        # Save to JSON
        output_path = output_dir / f"split_fold_{i}.json"
        try:
            with output_path.open("w") as f:
                json.dump({"train": train, "test": test}, f, indent=2)
            print(f"Created {output_path}")
        except IOError as e:
            print(f"Error writing {output_path}: {str(e)}")

if __name__ == "__main__":
    main()