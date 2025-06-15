"""
CARS Microscopy Image Dataset for Thyroid Classification
Handles 512x512 single-channel uint16 images
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import cv2
from PIL import Image
# import pandas as pd # No longer used directly in this file after refactor
from sklearn.model_selection import train_test_split # StratifiedKFold no longer used directly here
import hashlib # No longer used
import json

# Rich imports for beautiful progress bars
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Import our console utilities
try:
    from src.utils.logging import console, create_progress_bar # print_data_summary no longer used here
except ImportError:
    # Fallback if running standalone
    from rich.console import Console
    console = Console()
    
    def create_progress_bar(description: str = "Processing"):
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )

from src.config.schemas import DatasetConfig, TrainingConfig # Added TrainingConfig for create_data_loaders


class CARSThyroidDataset(Dataset):
    """
    PyTorch Dataset for CARS Thyroid Microscopy Images.
    Uses DatasetConfig for configuration.
    
    Args:
        config: DatasetConfig object containing all dataset parameters.
        mode: One of 'train', 'val', 'test', or 'all'.
        transform: Optional transform to be applied on images.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        mode: str = 'train',
        transform: Optional[Callable] = None,
    ):
        self.config = config
        self.mode = mode
        self.transform = transform
        
        self.image_paths: np.ndarray = np.array([]) # Will store paths for the current split
        self.labels: np.ndarray = np.array([])    # Will store labels for the current split
        self.indices: np.ndarray = np.array([])   # Will be np.arange(len(self.image_paths))
        
        # Determine and ensure splits_dir exists
        if not self.config.split_dir:
            data_root = Path(self.config.data_path)
            self.splits_dir = data_root.parent / 'splits'
            console.print(f"[yellow]Warning: config.split_dir not set, defaulting to {self.splits_dir}[/yellow]")
        else:
            self.splits_dir = Path(self.config.split_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_split_file: Optional[Path] = None # For reference, set by _load_split_data

        self._load_split_data()

    def _get_all_image_metadata(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scans the data directory and returns all image paths, labels, and patient IDs.
        """
        data_root = Path(self.config.data_path)
        image_paths_list = []
        labels_list = []
        patient_ids_list = []
        
        for class_idx, class_name in enumerate(['normal', 'cancerous']):
            class_dir = data_root / class_name
            if not class_dir.exists():
                continue
            
            supported_formats = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
            class_images = [p for fmt in supported_formats for p in class_dir.glob(fmt)]
            
            for img_path in class_images:
                image_paths_list.append(img_path)
                labels_list.append(class_idx)
                if '_' in img_path.stem and img_path.stem.split('_')[-1].isdigit():
                    patient_ids_list.append(f"{class_name}_{img_path.stem.split('_')[-1]}")
                else:
                    patient_ids_list.append(img_path.stem)
        
        if not image_paths_list:
            console.print(f"[red]Error: No images found in {data_root} or its subdirectories ('normal', 'cancerous'). Check config.data_path.[/red]")
        
        return np.array(image_paths_list), np.array(labels_list), np.array(patient_ids_list)

    def _generate_splits(self, all_image_paths: np.ndarray, all_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates train/val/test splits from all available data.
        Returns a dictionary of indices: {'train': ndarray, 'val': ndarray, 'test': ndarray}.
        """
        if len(all_image_paths) == 0:
            console.print("[yellow]Cannot generate splits: no images provided to _generate_splits.[/yellow]")
            return {'train': np.array([]), 'val': np.array([]), 'test': np.array([])}

        indices = np.arange(len(all_image_paths))
        
        test_ratio = self.config.test_split_ratio if self.config.test_split_ratio is not None else 0.15
        if not (0 < test_ratio < 1):
            console.print(f"[yellow]Warning: Invalid config.test_split_ratio ({self.config.test_split_ratio}). Using default 0.15.[/yellow]")
            test_ratio = 0.15

        val_ratio_of_train_val = self.config.val_split_ratio
        if not (0 < val_ratio_of_train_val < 1):
            console.print(f"[yellow]Warning: Invalid config.val_split_ratio ({self.config.val_split_ratio}). Using default 0.2.[/yellow]")
            val_ratio_of_train_val = 0.2

        stratify_by = all_labels if len(np.unique(all_labels)) > 1 else None

        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, stratify=stratify_by, random_state=self.config.random_seed
        )
        
        stratify_train_val_by = all_labels[train_val_indices] if stratify_by is not None and len(train_val_indices) > 0 else None
        if stratify_train_val_by is not None and len(np.unique(stratify_train_val_by)) < 2:
            stratify_train_val_by = None

        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_of_train_val, stratify=stratify_train_val_by, random_state=self.config.random_seed
        )
        
        generated_split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        }
        # Pass all_labels for summary context
        self._print_split_summary_from_indices(generated_split_indices, all_labels, "Data Split Summary (Generated)")
        return generated_split_indices

    def _print_split_summary_from_indices(self, splits_dict_indices: Dict[str, np.ndarray], all_dataset_labels: np.ndarray, title: str):
        """Prints a summary of data splits given a dictionary of indices and all labels."""
        if len(all_dataset_labels) == 0 and not any(len(v) > 0 for v in splits_dict_indices.values()):
             console.print(f"[yellow]{title}: No data to summarize.[/yellow]")
             return

        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Total", style="white")
        table.add_column("Normal", style="green")
        table.add_column("Cancerous", style="red")
        table.add_column("Percentage", style="yellow")

        # If all_dataset_labels is empty but splits_dict_indices has items (e.g. from k-fold file with paths)
        # then total_dataset_images for percentage calculation needs to be sum of split items.
        # However, for non-kfold generation, all_dataset_labels is the source.
        # For k-fold, we print summary based on loaded split, so all_dataset_labels is not directly used for total.
        
        # Let's refine total for percentage. If all_dataset_labels is available (non-kfold generation), use it.
        # Otherwise (k-fold from file), sum of current fold's splits.
        # For now, this method is primarily for generated splits, so all_dataset_labels is key.
        total_images_for_percentage = len(all_dataset_labels) if len(all_dataset_labels) > 0 else sum(len(v) for v in splits_dict_indices.values())


        for split_name, indices_list in splits_dict_indices.items():
            count = len(indices_list)
            if count == 0:
                table.add_row(split_name.capitalize(), "0", "0", "0", "0.0%")
                continue
            
            # If indices_list are actual indices into all_dataset_labels
            if len(all_dataset_labels) > 0 : # This implies non-kfold generation context
                split_labels = all_dataset_labels[indices_list]
                normal_count = (split_labels == 0).sum()
                cancer_count = (split_labels == 1).sum()
            else: # This implies k-fold context where indices_list might be self.labels directly
                  # This method is called _print_split_summary_from_INDICES, so it expects indices.
                  # For k-fold, we'd pass self.labels of the fold as 'all_dataset_labels' for this method.
                  # Or, this method is only for non-kfold generation summary.
                  # Let's assume for now it's for non-kfold generation.
                normal_count = -1 # Mark as N/A if all_dataset_labels not provided for context
                cancer_count = -1


            percentage = (count / total_images_for_percentage) * 100 if total_images_for_percentage > 0 else 0
            
            table.add_row(
                split_name.capitalize(),
                str(count),
                str(normal_count) if normal_count != -1 else "N/A",
                str(cancer_count) if cancer_count != -1 else "N/A",
                f"{percentage:.1f}%"
            )
        console.print(table)


    def _load_split_data(self):
        """
        Loads image paths and labels for the current mode, handling k-fold and standard splits.
        Populates self.image_paths, self.labels, and self.indices.
        """
        # Determine if a specific split file is provided directly in the config
        # This takes precedence for k-fold if KFoldExperiment sets it.
        if hasattr(self.config, 'split_file') and self.config.split_file:
            split_file_path = Path(self.config.split_file)
            self.current_split_file = split_file_path
            fold_num_for_log = self.config.fold if self.config.fold is not None else "N/A"
            console.print(f"[cyan]Loading specific split file (Fold {fold_num_for_log}, Mode '{self.mode}'):[/cyan] {split_file_path}")

            if not split_file_path.exists():
                msg = f"Provided split_file not found: {split_file_path}"
                console.print(f"[red]Error: {msg}[/red]")
                raise FileNotFoundError(msg)
            
            all_image_paths_np, all_labels_np, _ = self._get_all_image_metadata()
            if len(all_image_paths_np) == 0:
                console.print(f"[yellow]Warning: No images found by _get_all_image_metadata. Dataset for mode '{self.mode}' using split file {split_file_path} will be empty if indices are out of bounds or file is empty.[/yellow]")
                # Proceed to load indices, error will be caught if indices are invalid for empty all_image_paths_np

            current_mode_indices_from_file: List[int] = []
            try:
                with open(split_file_path, 'r') as f:
                    all_splits_indices_for_fold = json.load(f)
                
                if self.mode not in all_splits_indices_for_fold:
                    msg = f"Mode '{self.mode}' not found as a key in split file {split_file_path}. Available keys: {list(all_splits_indices_for_fold.keys())}"
                    console.print(f"[red]Error: {msg}[/red]")
                    raise ValueError(msg)
                
                current_mode_indices_from_file = all_splits_indices_for_fold[self.mode]
                console.print(f"[green]Loaded {len(current_mode_indices_from_file)} indices for mode '{self.mode}' from {split_file_path}.[/green]")

            except json.JSONDecodeError:
                console.print(f"[red]Error: Error decoding JSON from split file: {split_file_path}[/red]")
                raise ValueError(f"Error decoding JSON from split file: {split_file_path}")
            except TypeError:
                console.print(f"[red]Error: Split file {split_file_path} for mode '{self.mode}' does not contain a list of integers as expected.[/red]")
                raise ValueError(f"Split file {split_file_path} for mode '{self.mode}' has incorrect data format.")

            if not all(isinstance(idx, int) for idx in current_mode_indices_from_file):
                console.print(f"[red]Error: Split file {split_file_path} for mode '{self.mode}' contains non-integer values.[/red]")
                raise ValueError(f"Split file {split_file_path} for mode '{self.mode}' must contain only integers.")

            if current_mode_indices_from_file:
                if len(all_image_paths_np) == 0: # Cannot select from empty list
                     msg = f"Cannot apply indices from {split_file_path} as no source images were found (check data_path: {self.config.data_path})."
                     console.print(f"[red]Error: {msg}[/red]")
                     raise ValueError(msg)
                max_idx_from_file = np.max(current_mode_indices_from_file)
                if max_idx_from_file >= len(all_image_paths_np):
                    msg = (f"Invalid index {max_idx_from_file} found in split file {split_file_path} "
                           f"for mode '{self.mode}'. Max possible index is {len(all_image_paths_np) - 1}.")
                    console.print(f"[red]Error: {msg}[/red]")
                    raise ValueError(msg)
                
                self.image_paths = all_image_paths_np[current_mode_indices_from_file]
                self.labels = all_labels_np[current_mode_indices_from_file]
            else:
                self.image_paths = np.array([])
                self.labels = np.array([])
            
            summary_title = f"Dataset Summary (File: {split_file_path.name}, Mode: {self.mode})"
            split_counts = {"normal": (self.labels == 0).sum(), "cancerous": (self.labels == 1).sum(), "unknown": (self.labels == -1).sum()}
            console.print(Panel(
                f"Total: {len(self.labels)}\nNormal: {split_counts['normal']}\nCancerous: {split_counts['cancerous']}\nUnknown: {split_counts['unknown']}",
                title=summary_title, border_style="blue"
            ))

        elif self.config.use_kfold: # K-fold but split_file not directly provided, construct from fold number
            if self.config.fold is None:
                console.print("[red]Error: K-fold is enabled (use_kfold=True) but config.fold is not specified (and config.split_file is not set).[/red]")
                raise ValueError("K-fold is enabled, but 'fold' number is not specified and 'split_file' is not set.")

            all_image_paths_np, all_labels_np, _ = self._get_all_image_metadata()
            if len(all_image_paths_np) == 0:
                console.print(f"[yellow]Warning: No images found by _get_all_image_metadata. K-fold dataset for mode '{self.mode}', fold {self.config.fold} will be empty.[/yellow]")
                self.image_paths = np.array([])
                self.labels = np.array([])
                self.indices = np.array([])
                return

            split_file_prefix = self.config.split_file_prefix if hasattr(self.config, 'split_file_prefix') and self.config.split_file_prefix else "split_fold_"
            split_filename = f"{split_file_prefix}{self.config.fold}.json"
            split_file_path = self.splits_dir / split_filename
            self.current_split_file = split_file_path
            
            console.print(f"[cyan]K-fold active (Fold {self.config.fold}, Mode '{self.mode}'). Loading from constructed path:[/cyan] {split_file_path}")
            
            # ... (rest of the k-fold loading logic from original, using the constructed split_file_path) ...
            # This part is largely the same as the block above, just that split_file_path is constructed.
            # For brevity, assuming the logic from lines 240-287 of the original file is duplicated here,
            # operating on the `split_file_path` constructed immediately above.
            # Ensure to replace `fold {self.config.fold}` in logs with the actual fold number.
            current_mode_indices_from_file: List[int] = []
            try:
                with open(split_file_path, 'r') as f:
                    all_splits_indices_for_fold = json.load(f)
                
                if self.mode not in all_splits_indices_for_fold:
                    msg = f"Mode '{self.mode}' not found as a key in K-fold split file {split_file_path}. Available keys: {list(all_splits_indices_for_fold.keys())}"
                    console.print(f"[red]Error: {msg}[/red]")
                    raise ValueError(msg)
                
                current_mode_indices_from_file = all_splits_indices_for_fold[self.mode]
                console.print(f"[green]Loaded {len(current_mode_indices_from_file)} indices for mode '{self.mode}' from fold {self.config.fold} file.[/green]")

            except FileNotFoundError:
                console.print(f"[red]Error: K-fold split file not found: {split_file_path}[/red]")
                raise FileNotFoundError(f"K-fold split file not found: {split_file_path}")
            except json.JSONDecodeError:
                console.print(f"[red]Error: Error decoding JSON from K-fold split file: {split_file_path}[/red]")
                raise ValueError(f"Error decoding JSON from K-fold split file: {split_file_path}")
            except TypeError:
                console.print(f"[red]Error: K-fold split file {split_file_path} for mode '{self.mode}' does not contain a list of integers as expected.[/red]")
                raise ValueError(f"K-fold split file {split_file_path} for mode '{self.mode}' has incorrect data format.")

            if not all(isinstance(idx, int) for idx in current_mode_indices_from_file):
                console.print(f"[red]Error: K-fold split file {split_file_path} for mode '{self.mode}' contains non-integer values.[/red]")
                raise ValueError(f"K-fold split file {split_file_path} for mode '{self.mode}' must contain only integers.")

            if current_mode_indices_from_file:
                if len(all_image_paths_np) == 0:
                     msg = f"Cannot apply indices from {split_file_path} as no source images were found (check data_path: {self.config.data_path})."
                     console.print(f"[red]Error: {msg}[/red]")
                     raise ValueError(msg)
                max_idx_from_file = np.max(current_mode_indices_from_file)
                if max_idx_from_file >= len(all_image_paths_np):
                    msg = (f"Invalid index {max_idx_from_file} found in K-fold split file {split_file_path} "
                           f"for mode '{self.mode}'. Max possible index is {len(all_image_paths_np) - 1}.")
                    console.print(f"[red]Error: {msg}[/red]")
                    raise ValueError(msg)
                
                self.image_paths = all_image_paths_np[current_mode_indices_from_file]
                self.labels = all_labels_np[current_mode_indices_from_file]
            else:
                self.image_paths = np.array([])
                self.labels = np.array([])
            
            kfold_summary_title = f"K-Fold (Fold {self.config.fold}, Mode {self.mode}) Summary"
            kfold_split_counts = {"normal": (self.labels == 0).sum(), "cancerous": (self.labels == 1).sum(), "unknown": (self.labels == -1).sum()}
            console.print(Panel(
                f"Total: {len(self.labels)}\nNormal: {kfold_split_counts['normal']}\nCancerous: {kfold_split_counts['cancerous']}\nUnknown: {kfold_split_counts['unknown']}",
                title=kfold_summary_title, border_style="blue"
            ))

        else: # Standard non-k-fold loading
            all_image_paths_np, all_labels_np, _ = self._get_all_image_metadata()

            if len(all_image_paths_np) == 0 and self.mode != 'all': # 'all' can be empty if no data at all
                console.print(f"[yellow]Warning: No images found by _get_all_image_metadata. Dataset for mode '{self.mode}' will be empty.[/yellow]")
                self.image_paths = np.array([])
                self.labels = np.array([])
                self.indices = np.array([])
                return

            if self.mode == 'all':
                self.image_paths = all_image_paths_np
                self.labels = all_labels_np
                console.print(f"[cyan]Mode 'all': Loaded all {len(self.image_paths)} images.[/cyan]")
                # Print summary for 'all' mode if desired
                # self._print_split_summary_from_indices({'all': np.arange(len(all_labels_np))}, all_labels_np, "Overall Dataset Summary (Mode 'all')")

            else: # 'train', 'val', 'test' for non-kfold
                split_file_for_indices: Optional[Path] = None
                if self.mode == 'test':
                    test_specific_file = self.splits_dir / 'test_split.json'
                    general_split_file = self.splits_dir / 'split_info.json' # Fallback
                    if test_specific_file.exists():
                        split_file_for_indices = test_specific_file
                    elif general_split_file.exists():
                        split_file_for_indices = general_split_file
                        console.print(f"[yellow]Test specific split file '{test_specific_file}' not found. Using '{general_split_file}' for test mode.[/yellow]")
                    else: # Neither exists, will attempt to load test_specific_file to trigger error or generation if applicable
                        split_file_for_indices = test_specific_file
                        console.print(f"[yellow]Neither '{test_specific_file}' nor '{general_split_file}' found for test mode. Will attempt to load '{test_specific_file}'.[/yellow]")
                
                elif self.mode in ['train', 'val']:
                    split_file_for_indices = self.splits_dir / 'split_info.json'
                
                self.current_split_file = split_file_for_indices

                current_mode_indices: Optional[np.ndarray] = None
                loaded_from_file = False
                if split_file_for_indices and split_file_for_indices.exists():
                    console.print(f"[cyan]Standard mode '{self.mode}'. Loading indices from:[/cyan] {split_file_for_indices}")
                    try:
                        with open(split_file_for_indices, 'r') as f:
                            split_data_from_file = json.load(f)
                        if self.mode not in split_data_from_file:
                            console.print(f"[yellow]Warning: Mode '{self.mode}' not found in split file {split_file_for_indices}. Available keys: {list(split_data_from_file.keys())}[/yellow]")
                            # For test mode, if 'test' key is missing from 'test_split.json', it's an error.
                            # If 'test' key is missing from 'split_info.json' (when used as fallback), also an issue.
                            if self.mode == 'test':
                                raise KeyError(f"Mode '{self.mode}' not found in designated split file {split_file_for_indices}.")
                        else:
                            current_mode_indices = np.array(split_data_from_file[self.mode])
                            loaded_from_file = True
                            console.print(f"[green]Loaded {len(current_mode_indices)} indices for mode '{self.mode}' from {split_file_for_indices}.[/green]")
                    except (json.JSONDecodeError, KeyError) as e:
                        console.print(f"[red]Error loading or parsing split file {split_file_for_indices}: {e}[/red]")
                        if self.mode == 'test': # Critical for test mode
                            raise e
                
                if not loaded_from_file and self.mode != 'test': # Try generation for train/val if loading failed or file didn't exist
                    console.print(f"[yellow]Indices for mode '{self.mode}' not loaded from file. Attempting to generate default splits...[/yellow]")
                    # _generate_splits expects all_image_paths and all_labels
                    generated_splits_indices_dict = self._generate_splits(all_image_paths_np, all_labels_np)
                    
                    if self.mode in generated_splits_indices_dict:
                        current_mode_indices = generated_splits_indices_dict[self.mode]
                        # Save the generated splits to 'split_info.json'
                        output_split_file = self.splits_dir / 'split_info.json'
                        console.print(f"[cyan]Saving newly generated splits to:[/cyan] {output_split_file}")
                        with open(output_split_file, 'w') as f:
                            json_serializable_splits = {k: v.tolist() for k, v in generated_splits_indices_dict.items()}
                            json.dump(json_serializable_splits, f, indent=2)
                        self.current_split_file = output_split_file # Update current_split_file reference
                    else: # Should not happen if _generate_splits is correct
                        msg = f"Mode '{self.mode}' not found in generated splits."
                        console.print(f"[red]Error: {msg}[/red]")
                        raise ValueError(msg)
                
                elif not loaded_from_file and self.mode == 'test':
                     msg = f"Test indices for mode '{self.mode}' could not be loaded from {split_file_for_indices}, and generation is not performed for test mode by default."
                     console.print(f"[red]Error: {msg}[/red]")
                     raise FileNotFoundError(msg)


                if current_mode_indices is not None:
                    if len(all_image_paths_np) == 0 and len(current_mode_indices) > 0 :
                        msg = "No images loaded by _get_all_image_metadata, but split file provided indices. Check data_path."
                        console.print(f"[red]Error: {msg}[/red]")
                        raise ValueError(msg)
                    
                    if len(current_mode_indices) > 0 and np.max(current_mode_indices) >= len(all_image_paths_np):
                        msg = f"Invalid indices found in split file for mode '{self.mode}'. Max index {np.max(current_mode_indices)} vs {len(all_image_paths_np)} total images."
                        console.print(f"[red]Error: {msg}[/red]")
                        raise ValueError(msg)

                    self.image_paths = all_image_paths_np[current_mode_indices]
                    self.labels = all_labels_np[current_mode_indices]
                    # Print summary for the loaded non-kfold split
                    # self._print_split_summary_from_indices({self.mode: current_mode_indices}, all_labels_np, f"Standard Mode '{self.mode}' Summary (from indices)")

                else: # Should be caught by earlier errors if current_mode_indices is still None
                    msg = f"Failed to obtain indices for mode '{self.mode}' for non-kfold split."
                    console.print(f"[red]Error: {msg}[/red]")
                    self.image_paths = np.array([])
                    self.labels = np.array([])
        
        self.indices = np.arange(len(self.image_paths)) # Indices are always 0..N-1 for the current split

        if len(self.image_paths) == 0 and self.mode != 'all':
             console.print(f"[yellow]Warning: Dataset for mode '{self.mode}' is empty after loading splits.[/yellow]")
        elif len(self.image_paths) > 0:
             console.print(f"[blue]Successfully populated dataset for mode '{self.mode}' with {len(self.image_paths)} samples.[/blue]")
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load a single image from self.image_paths using the direct index for the current split."""
        
        if idx >= len(self.image_paths): # idx is already the direct index for the current split
            raise IndexError(f"Index {idx} is out of bounds for current split's image_paths (len: {len(self.image_paths)}).")

        img_path = self.image_paths[idx]
        
        # Load image based on format
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            img = tifffile.imread(str(img_path))
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None: # Try with PIL as a fallback
                try:
                    pil_img = Image.open(img_path)
                    img = np.array(pil_img)
                except Exception as e:
                    raise IOError(f"Failed to load image {img_path} with OpenCV and PIL: {e}")

        if img is None: # Should not happen if previous block works
             raise IOError(f"Failed to load image {img_path}")

        # Ensure single channel (as per DatasetConfig.channels, assumed 1 for CARS)
        if len(img.shape) == 3:
            if self.config.channels == 1:
                if img.shape[2] == 3: # RGB to Gray
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 4: # RGBA to Gray
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                elif img.shape[2] >= self.config.channels: # Take first channel if multi-channel but expecting 1
                    img = img[:, :, 0]
                else:
                    raise ValueError(f"Image {img_path} has {img.shape[2]} channels, expected {self.config.channels} or convertible (e.g. 3 for RGB).")
            # else: # If config.channels == 3, and image is 3-channel, it's fine.
            #    pass # Add logic for >1 channel if needed, e.g. ensuring correct order BGR vs RGB
        elif len(img.shape) == 2 and self.config.channels != 1:
            raise ValueError(f"Image {img_path} is 2D (grayscale) but config.channels is {self.config.channels}.")
        
        # Ensure uint16 (original CARS data type)
        if img.dtype != np.uint16:
            if img.dtype == np.uint8: # Common case for PNG/JPG
                img = img.astype(np.uint16) * 257  # Scale 0-255 to 0-65535
            else: # Other types, attempt direct conversion
                img = img.astype(np.uint16)
        
        return img
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize and normalize to [0,1] float32 tensor."""
        
        # Resize if needed
        target_h, target_w = self.config.img_size, self.config.img_size # Assuming square
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize uint16 to [0, 1] float32
        img_float = img.astype(np.float32) / 65535.0
        
        # Convert to tensor and add channel dimension (unsqueeze for single channel)
        # Shape: [C, H, W] where C is self.config.channels
        img_tensor = torch.from_numpy(img_float)
        if self.config.channels == 1 and len(img_tensor.shape) == 2: # H, W -> 1, H, W
            img_tensor = img_tensor.unsqueeze(0)
        # Add logic here if self.config.channels > 1, e.g. for H, W, C -> C, H, W
        
        return img_tensor
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample. idx is the direct index for the current split."""
        if idx >= len(self.image_paths): # self.image_paths now holds only the current split's data
             raise IndexError(f"Index {idx} is out of bounds for current split '{self.mode}' (len: {len(self.image_paths)}).")

        img = self._load_image(idx) # Pass idx directly
        img_tensor = self._preprocess_image(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # self.labels also holds only the current split's labels
        if idx >= len(self.labels):
            raise IndexError(f"Index {idx} is out of bounds for current split's labels (len: {len(self.labels)}). This indicates an internal inconsistency.")
        label = self.labels[idx]
        
        # Ensure label is torch.long for CrossEntropyLoss
        return img_tensor, torch.tensor(label, dtype=torch.long)
    
    def get_sample_batch(self, n_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of random samples for visualization."""
        if len(self) == 0:
            console.print("[yellow]Cannot get sample batch: dataset is empty.[/yellow]")
            return torch.empty(0), torch.empty(0)
        
        actual_n_samples = min(n_samples, len(self))
        indices = np.random.choice(len(self), actual_n_samples, replace=False)
        
        images = []
        labels_list = []
        
        for i in indices:
            img, label_val = self[i]
            images.append(img)
            labels_list.append(label_val)
        
        return torch.stack(images), torch.tensor(labels_list)


def create_data_loaders(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig, # For batch_size, num_workers
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    # fold parameter is now part of dataset_config
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders using DatasetConfig.
    
    Args:
        dataset_config: Configuration for the dataset.
        training_config: Configuration for training parameters like batch_size, num_workers.
        transform_train: Transform for training data.
        transform_val: Transform for validation/test data.
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders.
    """
    
    console.print(Panel.fit(
        "[bold cyan]Creating CARS Thyroid Data Loaders (Config-Driven)[/bold cyan]",
        border_style="blue"
    ))
    
    dataloaders = {}
    
    for split_mode in ['train', 'val', 'test']:
        transform = transform_train if split_mode == 'train' else transform_val
        
        # The same dataset_config is passed; CARSThyroidDataset uses its 'mode' and internal
        # config (like 'fold' if use_kfold is true) to load the correct data.
        dataset = CARSThyroidDataset(
            config=dataset_config,
            mode=split_mode,
            transform=transform,
        )
        
        if len(dataset) == 0:
            console.print(f"[yellow]Warning: Dataset for split '{split_mode}' is empty. DataLoader will also be empty.[/yellow]")
        
        dataloaders[split_mode] = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=(split_mode == 'train'),
            num_workers=training_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_mode == 'train' and len(dataset) > 0) # drop_last only if shuffle and not empty
        )
    
    # Print summary
    # Calculate total images carefully, as one split might be empty
    train_len = len(dataloaders['train'].dataset)
    val_len = len(dataloaders['val'].dataset)
    test_len = len(dataloaders['test'].dataset)
    
    console.print(f"\n[bold]Dataset Summary (from DataLoaders):[/bold]")
    # Total unique images in the source data, not sum of splits if k-fold
    # This requires CARSThyroidDataset to expose total unique images if needed.
    # For now, sum of current split lengths:
    console.print(f"Image size: ({dataset_config.img_size}, {dataset_config.img_size}, {dataset_config.channels})")
    console.print(f"Splits (lengths):")
    console.print(f"  Train: {train_len} images")
    console.print(f"  Val: {val_len} images")
    console.print(f"  Test: {test_len} images")
    
    return dataloaders


# Demo function
def demo_dataset():
    """Demo the dataset functionality using DatasetConfig."""
    
    # Define a sample DatasetConfig
    # Ensure data_path points to a directory with 'normal' and 'cancerous' subfolders
    # and split_dir points to where split JSONs are or will be created.
    sample_data_path = Path("data/raw_demo") # Use a dedicated demo path
    sample_splits_path = Path("data/splits_demo")

    # Create sample data structure if it doesn't exist for the demo
    if not sample_data_path.exists():
        console.print(f"[yellow]Creating sample data structure for demo at {sample_data_path}...[/yellow]")
        for class_name in ['normal', 'cancerous']:
            (sample_data_path / class_name).mkdir(parents=True, exist_ok=True)
            # Create a dummy .tif file for testing
            dummy_image = np.zeros((32, 32), dtype=np.uint16) # Small dummy image
            tifffile.imwrite(sample_data_path / class_name / f"dummy_sample_1.tif", dummy_image)
            tifffile.imwrite(sample_data_path / class_name / f"dummy_sample_2.tif", dummy_image)
        sample_splits_path.mkdir(parents=True, exist_ok=True)
        console.print("[green]✓ Created sample directories and dummy images for demo.[/green]")
        console.print(f"[yellow]Demo will use data from: {sample_data_path}[/yellow]")
        console.print(f"[yellow]Demo will use/create splits in: {sample_splits_path}[/yellow]")

    try:
        dataset_cfg = DatasetConfig(
            name="cars_thyroid_demo",
            data_path=str(sample_data_path),
            split_dir=str(sample_splits_path),
            use_kfold=False, # For demo, let's use simple split generation
            fold=None,
            val_split_ratio=0.25, # 25% of (train+val) for validation
            test_split_ratio=0.2,  # 20% of total for test
            img_size=32, # Match dummy image size for speed
            channels=1,
            mean=[0.5], # Example
            std=[0.5],  # Example
            apply_augmentations=False,
            quality_preprocessing=False
        )

        # Create dataset for 'train' mode
        train_dataset = CARSThyroidDataset(config=dataset_cfg, mode='train')
        
        if len(train_dataset) > 0:
            img, label = train_dataset[0]
            console.print(f"\n[cyan]Sample from 'train' dataset:[/cyan]")
            console.print(f"  Image shape: {img.shape}")
            console.print(f"  Label: {label} ({'normal' if label == 0 else 'cancerous'})")
            console.print(f"  Data type: {img.dtype}")
            console.print(f"  Value range: [{img.min():.3f}, {img.max():.3f}]")
        else:
            console.print("[yellow]Train dataset is empty. Check data loading and split generation.[/yellow]")

        # Create data loaders
        # Define a sample TrainingConfig for batch_size and num_workers
        training_cfg = TrainingConfig(
            batch_size=2, # Small batch for demo
            num_workers=0 # Simpler for demo, avoid multiprocessing issues
        )

        dataloaders = create_data_loaders(
            dataset_config=dataset_cfg,
            training_config=training_cfg
        )
        
        console.print(f"\n[cyan]DataLoaders created for modes: {list(dataloaders.keys())}[/cyan]")
        if len(dataloaders['train']) > 0:
            train_batch_images, train_batch_labels = next(iter(dataloaders['train']))
            console.print(f"  Sample train batch images shape: {train_batch_images.shape}")
            console.print(f"  Sample train batch labels: {train_batch_labels}")
        else:
            console.print("[yellow]Train DataLoader is empty.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during demo_dataset: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())