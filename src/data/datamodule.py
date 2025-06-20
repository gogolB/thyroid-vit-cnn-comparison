# src/data/datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from src.config.schemas import DatasetConfig, TrainingConfig
from src.data.dataset import CARSThyroidDataset
from src.data.quality_preprocessing import create_quality_aware_transform # Assuming this exists/will be created
# from src.data.transforms import get_transforms # Assuming this exists/will be created

# Placeholder for get_transforms if not available yet for initial implementation
# You might need to create/adapt src/data/transforms.py
import torchvision.transforms as T
def get_transforms(dataset_config: DatasetConfig, mode: str):
    transforms_list = []
    # Ensure img_size is an int if it's coming from a config that might be a float
    img_size = int(dataset_config.img_size)

    if dataset_config.channels == 1:
        transforms_list.append(T.Grayscale(num_output_channels=1))
    
    transforms_list.append(T.Resize((img_size, img_size)))
    
    if mode == 'train' and dataset_config.apply_augmentations:
        # Add some basic augmentations
        transforms_list.append(T.RandomHorizontalFlip())
        transforms_list.append(T.RandomRotation(10))
        # Add more as needed based on project requirements
        
    # T.ToTensor() is removed as _preprocess_image in CARSThyroidDataset already returns a tensor
    transforms_list.append(T.Normalize(mean=dataset_config.mean, std=dataset_config.std))
    return T.Compose(transforms_list)


class ThyroidDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config: DatasetConfig, training_config: TrainingConfig):
        super().__init__()
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.save_hyperparameters(logger=False) # Good practice

        # Transforms will be initialized in setup
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        # Datasets will be initialized in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None # For predict stage

    def prepare_data(self):
        # Called only on 1 GPU
        # Download, tokenize, etc.
        # For CARS, if splits need to be generated and saved, this could be a place.
        # For now, assuming CARSThyroidDataset handles its own data loading/split logic.
        pass

    def setup(self, stage: Optional[str] = None):
        # Called on every GPU
        # Assign train/val/test datasets for use in dataloaders
        
        # Setup transforms
        self.train_transforms = create_quality_aware_transform(self.dataset_config.img_size, Path("reports/quality_report.json"), augmentation_level="medium", split="train")
        self.val_transforms = create_quality_aware_transform(self.dataset_config.img_size, Path("reports/quality_report.json"), augmentation_level="none", split="val")
        self.test_transforms = create_quality_aware_transform(self.dataset_config.img_size, Path("reports/quality_report.json"), augmentation_level="none", split="test")

        if stage == 'fit' or stage is None:
            self.train_dataset = CARSThyroidDataset(
                config=self.dataset_config, 
                mode='train', 
                transform=self.train_transforms
            )
            self.val_dataset = CARSThyroidDataset(
                config=self.dataset_config, 
                mode='val', 
                transform=self.val_transforms
            )
            if not self.train_dataset or not hasattr(self.train_dataset, 'image_paths') or self.train_dataset.image_paths.size == 0:
                raise ValueError("Training dataset is empty or not initialized correctly. Check data paths and split configurations.")
            if not self.val_dataset or not hasattr(self.val_dataset, 'image_paths') or self.val_dataset.image_paths.size == 0:
                raise ValueError("Validation dataset is empty or not initialized correctly. Check data paths and split configurations.")

        if stage == 'test' or stage is None:
            self.test_dataset = CARSThyroidDataset(
                config=self.dataset_config, 
                mode='test', 
                transform=self.test_transforms
            )
            # Optional: Check if test_dataset is empty and print a warning
            # if not self.test_dataset or not hasattr(self.test_dataset, 'image_paths') or not self.test_dataset.image_paths:
            #     print("Warning: Test dataset is empty or not initialized correctly.")

        if stage == 'predict' or stage is None:
            # Typically, prediction uses test transforms and test data, or a specific predict set
            self.predict_dataset = CARSThyroidDataset(
                config=self.dataset_config,
                mode='test', # Or a specific 'predict' mode if your dataset supports it
                transform=self.test_transforms # Or specific predict_transforms
            )
            # Optional: Check if predict_dataset is empty
            # if not self.predict_dataset or not hasattr(self.predict_dataset, 'image_paths') or not self.predict_dataset.image_paths:
            #     print("Warning: Predict dataset is empty or not initialized correctly.")


    def train_dataloader(self):
        if not self.train_dataset:
            # Attempt to set up if called before explicit setup (e.g. during auto-lr-find)
            if 'fit' not in self._has_setup_fit and self._trainer is not None:
                 self.setup(stage='fit')
            if not self.train_dataset: # Re-check after attempting setup
                raise RuntimeError("Train dataset not setup. Call setup('fit') first or ensure trainer calls it.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.dataset_config.num_workers > 0 else False
        )

    def val_dataloader(self):
        if not self.val_dataset:
            if 'fit' not in self._has_setup_fit and self._trainer is not None:
                self.setup(stage='fit')
            if not self.val_dataset:
                raise RuntimeError("Validation dataset not setup. Call setup('fit') first or ensure trainer calls it.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.dataset_config.num_workers > 0 else False
        )

    def test_dataloader(self):
        if not self.test_dataset:
            if 'test' not in self._has_setup_test and self._trainer is not None:
                self.setup(stage='test')
            if not self.test_dataset: # Re-check after attempting setup
                 # It's possible test_dataset might be empty if no test split is defined.
                 # Depending on requirements, could return None or raise error
                 print("Warning: Test dataset is empty after attempting setup in test_dataloader. Returning None.")
                 return None # Or an empty DataLoader
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.dataset_config.num_workers > 0 else False
        )

    def predict_dataloader(self):
        if not self.predict_dataset:
            if 'predict' not in self._has_setup_predict and self._trainer is not None:
                self.setup(stage='predict')
            if not self.predict_dataset:
                print("Warning: Predict dataset not setup. Call setup('predict') first or ensure trainer calls it. Returning None.")
                return None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.dataset_config.num_workers > 0 else False
        )