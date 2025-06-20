import pytest
import numpy as np
import torch
import tifffile
import cv2
from PIL import Image
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call

from src.data.dataset import CARSThyroidDataset, create_data_loaders
from src.config.schemas import DatasetConfig, TrainingConfig

# Rich console is used for printing, mock it to avoid output during tests
@pytest.fixture(autouse=True)
def mock_rich_console():
    with patch('src.data.dataset.console', MagicMock()) as mock_console:
        # If create_progress_bar is also used directly from dataset.py and not just via src.utils.logging
        with patch('src.data.dataset.create_progress_bar', return_value=MagicMock()):
            yield mock_console

@pytest.fixture
def base_dataset_config(tmp_path):
    """Provides a base DatasetConfig pointing to a temporary data path."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    (data_dir / "normal").mkdir()
    (data_dir / "cancerous").mkdir()
    
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    return DatasetConfig(
        name="test_dataset",
        data_path=str(data_dir),
        split_dir=str(splits_dir),
        use_kfold=False,
        fold=None,
        val_split_ratio=0.2,
        test_split_ratio=0.15,
        img_size=32,
        channels=1,
        mean=[0.5],
        std=[0.5],
        apply_augmentations=False,
        quality_preprocessing=False
    )

@pytest.fixture
def base_training_config():
    """Provides a base TrainingConfig."""
    return TrainingConfig(
        batch_size=2,
        num_workers=0
    )

@pytest.fixture
def create_dummy_image_files(base_dataset_config):
    """Creates dummy image files in the data directory specified by base_dataset_config."""
    data_path = Path(base_dataset_config.data_path)
    dummy_img_data = np.zeros((32, 32), dtype=np.uint16)
    
    # Normal images
    tifffile.imwrite(data_path / "normal" / "norm1_patient1.tif", dummy_img_data)
    tifffile.imwrite(data_path / "normal" / "norm2_patient2.tif", dummy_img_data)
    cv2.imwrite(str(data_path / "normal" / "norm3_patient3.png"), dummy_img_data.astype(np.uint8)) # Save as uint8 for png
    
    # Cancerous images
    tifffile.imwrite(data_path / "cancerous" / "canc1_patientA.tif", dummy_img_data)
    tifffile.imwrite(data_path / "cancerous" / "canc2_patientB.tiff", dummy_img_data)
    # Create a JPG image (requires 3 channels or grayscale)
    pil_img_gray = Image.fromarray(dummy_img_data.astype(np.uint8), mode='L')
    pil_img_gray.save(data_path / "cancerous" / "canc3_patientC.jpg")

    return {
        "normal": [
            data_path / "normal" / "norm1_patient1.tif",
            data_path / "normal" / "norm2_patient2.tif",
            data_path / "normal" / "norm3_patient3.png"
        ],
        "cancerous": [
            data_path / "cancerous" / "canc1_patientA.tif",
            data_path / "cancerous" / "canc2_patientB.tiff",
            data_path / "cancerous" / "canc3_patientC.jpg"
        ]
    }

# --- CARSThyroidDataset Tests ---

class TestCARSThyroidDatasetInit:
    def test_init_default_split_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = DatasetConfig(data_path=str(data_dir), split_dir=None) # No split_dir
        
        with patch.object(CARSThyroidDataset, '_load_split_data') as mock_load_split:
            dataset = CARSThyroidDataset(config=config, mode='train')
            expected_splits_dir = Path(data_dir).parent / 'splits'
            assert dataset.splits_dir == expected_splits_dir
            assert expected_splits_dir.exists()
            mock_load_split.assert_called_once()

    def test_init_provided_split_dir(self, base_dataset_config):
        splits_dir = Path(base_dataset_config.split_dir)
        assert splits_dir.exists() # Ensure fixture created it
        
        with patch.object(CARSThyroidDataset, '_load_split_data') as mock_load_split:
            dataset = CARSThyroidDataset(config=base_dataset_config, mode='train')
            assert dataset.splits_dir == splits_dir
            mock_load_split.assert_called_once()
            assert dataset.config == base_dataset_config
            assert dataset.mode == 'train'
            assert dataset.transform is None

    def test_init_with_transform(self, base_dataset_config):
        mock_transform = MagicMock()
        with patch.object(CARSThyroidDataset, '_load_split_data'):
            dataset = CARSThyroidDataset(config=base_dataset_config, mode='val', transform=mock_transform)
            assert dataset.transform == mock_transform


class TestCARSThyroidDatasetGetAllImageMetadata:
    def test_get_all_image_metadata_success(self, base_dataset_config, create_dummy_image_files):
        dataset = CARSThyroidDataset(config=base_dataset_config, mode='all') # Mode 'all' to trigger metadata scan
        
        # The _load_split_data in __init__ calls _get_all_image_metadata.
        # We can inspect the results if mode='all' populates image_paths and labels directly,
        # or call it directly for more isolated testing.
        
        # For this test, let's call it directly to isolate.
        # We need an instance, but can prevent _load_split_data from running in __init__
        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        
        paths, labels, p_ids = instance._get_all_image_metadata()
        
        expected_total_images = len(create_dummy_image_files["normal"]) + len(create_dummy_image_files["cancerous"])
        assert len(paths) == expected_total_images
        assert len(labels) == expected_total_images
        assert len(p_ids) == expected_total_images

        # Check if paths are Path objects and exist
        for p in paths:
            assert isinstance(p, Path)
            assert p.exists()

        # Check labels (0 for normal, 1 for cancerous)
        # Order might vary based on glob, so check counts
        normal_count = sum(1 for p in paths if "normal" in str(p.parent.name))
        cancerous_count = sum(1 for p in paths if "cancerous" in str(p.parent.name))
        
        assert np.sum(labels == 0) == normal_count
        assert np.sum(labels == 1) == cancerous_count
        
        # Check patient IDs (simplified check)
        assert "norm1_patient1" in p_ids # Adjusted to match actual ID generation
        assert "canc1_patientA" in p_ids # Adjusted to match actual ID generation
        assert "norm3_patient3" in p_ids # No underscore number, uses stem

    def test_get_all_image_metadata_no_images(self, base_dataset_config):
        # Config points to an empty data_path (or one without normal/cancerous subdirs with images)
        empty_data_dir = base_dataset_config.data_path # Fixture creates subdirs, but they are empty initially
        
        # Clear any dummy files if created by other fixtures for this specific config instance
        for item in (Path(empty_data_dir) / "normal").iterdir(): item.unlink()
        for item in (Path(empty_data_dir) / "cancerous").iterdir(): item.unlink()

        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        
        paths, labels, p_ids = instance._get_all_image_metadata()
        
        assert len(paths) == 0
        assert len(labels) == 0
        assert len(p_ids) == 0
        # Check for console print warning (mock_rich_console is active)
        # dataset.console.print.assert_any_call(f"[red]Error: No images found in {Path(base_dataset_config.data_path)} or its subdirectories ('normal', 'cancerous'). Check config.data_path.[/red]")


    def test_get_all_image_metadata_one_class_missing(self, base_dataset_config, create_dummy_image_files):
        # Remove 'cancerous' images to simulate a missing class directory or empty class
        cancerous_dir = Path(base_dataset_config.data_path) / "cancerous"
        for item in cancerous_dir.iterdir():
            item.unlink()
        cancerous_dir.rmdir() # Remove the directory itself

        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        paths, labels, p_ids = instance._get_all_image_metadata()

        expected_normal_images = len(create_dummy_image_files["normal"])
        assert len(paths) == expected_normal_images
        assert len(labels) == expected_normal_images
        assert all(l == 0 for l in labels) # All should be normal


class TestCARSThyroidDatasetGenerateSplits:
    @pytest.fixture
    def mock_train_test_split(self):
        with patch('src.data.dataset.train_test_split') as mock_split:
            # Setup a default behavior for mock_split
            # e.g., mock_split.side_effect = lambda data, test_size, stratify, random_state: (np.array([0,1]), np.array([2]))
            yield mock_split

    def test_generate_splits_basic(self, base_dataset_config, mock_train_test_split):
        all_image_paths = np.array([Path(f"img_{i}.tif") for i in range(10)])
        all_labels = np.array([0]*5 + [1]*5) # Balanced labels

        # Define side effect for train_test_split
        # First call (train_val vs test)
        # 10 items, test_ratio=0.15 (config default) -> 1.5, so 2 test items, 8 train_val items
        # Let's say indices are 0..9. Test: [8,9], Train_val: [0,1,2,3,4,5,6,7]
        # Second call (train vs val on train_val_indices)
        # 8 items, val_ratio_of_train_val=0.2 (config default) -> 1.6, so 2 val items, 6 train items
        # Let's say from [0..7], Val: [6,7], Train: [0,1,2,3,4,5]
        
        # These are indices *within the passed array* to train_test_split
        mock_train_test_split.side_effect = [
            (np.array([0,1,2,3,4,5,6,7]), np.array([8,9])), # train_val_indices, test_indices
            (np.array([0,1,2,3,4,5]), np.array([6,7]))      # train_indices, val_indices (relative to train_val_indices)
        ]

        with patch.object(CARSThyroidDataset, '_load_split_data'): # Prevent __init__ from running full load
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        
        with patch.object(instance, '_print_split_summary_from_indices') as mock_print_summary:
            splits = instance._generate_splits(all_image_paths, all_labels)

        assert mock_train_test_split.call_count == 2
        
        # First call to train_test_split
        args1, kwargs1 = mock_train_test_split.call_args_list[0]
        assert np.array_equal(args1[0], np.arange(10)) # indices
        assert kwargs1['test_size'] == base_dataset_config.test_split_ratio
        assert np.array_equal(kwargs1['stratify'], all_labels)
        assert kwargs1['random_state'] == base_dataset_config.random_seed

        # Second call to train_test_split
        args2, kwargs2 = mock_train_test_split.call_args_list[1]
        assert np.array_equal(args2[0], np.array([0,1,2,3,4,5,6,7])) # train_val_indices
        assert kwargs2['test_size'] == base_dataset_config.val_split_ratio
        # Stratify labels for the second split should be all_labels[train_val_indices]
        assert np.array_equal(kwargs2['stratify'], all_labels[np.array([0,1,2,3,4,5,6,7])])
        assert kwargs2['random_state'] == base_dataset_config.random_seed

        assert np.array_equal(splits['train'], np.array([0,1,2,3,4,5]))
        assert np.array_equal(splits['val'], np.array([6,7]))
        assert np.array_equal(splits['test'], np.array([8,9]))
        mock_print_summary.assert_called_once()

    def test_generate_splits_invalid_ratios(self, base_dataset_config, mock_train_test_split):
        base_dataset_config.test_split_ratio = -0.1 # Invalid
        base_dataset_config.val_split_ratio = 1.5  # Invalid

        all_image_paths = np.array([Path(f"img_{i}.tif") for i in range(10)])
        all_labels = np.array([0]*5 + [1]*5)

        mock_train_test_split.side_effect = [
            (np.arange(8), np.arange(8,10)), # Dummy return for 1st call (0.15 test ratio)
            (np.arange(6), np.arange(6,8))   # Dummy return for 2nd call (0.2 val ratio)
        ]
        
        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        with patch.object(instance, '_print_split_summary_from_indices'):
            instance._generate_splits(all_image_paths, all_labels)

        # Check that default ratios were used in train_test_split calls
        _, kwargs1 = mock_train_test_split.call_args_list[0]
        assert kwargs1['test_size'] == 0.15 # Default
        _, kwargs2 = mock_train_test_split.call_args_list[1]
        assert kwargs2['test_size'] == 0.2  # Default

    def test_generate_splits_single_label_no_stratify(self, base_dataset_config, mock_train_test_split):
        all_image_paths = np.array([Path(f"img_{i}.tif") for i in range(10)])
        all_labels = np.array([0]*10) # Single class

        mock_train_test_split.side_effect = [
            (np.arange(8), np.arange(8,10)), 
            (np.arange(6), np.arange(6,8))
        ]
        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        with patch.object(instance, '_print_split_summary_from_indices'):
            instance._generate_splits(all_image_paths, all_labels)
        
        _, kwargs1 = mock_train_test_split.call_args_list[0]
        assert kwargs1['stratify'] is None # Stratify should be None if only one class
        
        # Second call might still get labels if train_val_indices is not empty,
        # but if those labels are all the same, stratify should also be None.
        _, kwargs2 = mock_train_test_split.call_args_list[1]
        assert kwargs2['stratify'] is None


    def test_generate_splits_empty_input(self, base_dataset_config, mock_train_test_split):
        all_image_paths = np.array([])
        all_labels = np.array([])
        with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        with patch.object(instance, '_print_split_summary_from_indices'):
            splits = instance._generate_splits(all_image_paths, all_labels)

        assert mock_train_test_split.call_count == 0
        assert np.array_equal(splits['train'], np.array([]))
        assert np.array_equal(splits['val'], np.array([]))
        assert np.array_equal(splits['test'], np.array([]))

# More tests for _load_split_data, _load_image, _preprocess_image, __getitem__, __len__, get_sample_batch
# And for create_data_loaders

class TestCARSThyroidDatasetLoadImage:
    @pytest.fixture
    def dataset_instance_for_load_image(self, base_dataset_config, tmp_path):
        # Create a dummy image file
        img_path = tmp_path / "test_image.tif"
        tifffile.imwrite(img_path, np.zeros((32,32), dtype=np.uint16) + 1000) # Value 1000

        png_path = tmp_path / "test_image.png"
        cv2.imwrite(str(png_path), np.zeros((32,32), dtype=np.uint8) + 50) # Value 50

        rgb_png_path = tmp_path / "rgb_image.png"
        cv2.imwrite(str(rgb_png_path), np.full((32,32,3), [10,20,30], dtype=np.uint8))


        with patch.object(CARSThyroidDataset, '_load_split_data'): # Prevent full init
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
        instance.image_paths = np.array([img_path, png_path, rgb_png_path]) # Manually set image_paths
        return instance, img_path, png_path, rgb_png_path

    def test_load_image_tif(self, dataset_instance_for_load_image):
        instance, tif_path, _, _ = dataset_instance_for_load_image
        img = instance._load_image(0) # First image is tif
        assert img.shape == (32, 32)
        assert img.dtype == np.uint16
        assert img[0,0] == 1000

    def test_load_image_png_uint8_to_uint16(self, dataset_instance_for_load_image):
        instance, _, png_path, _ = dataset_instance_for_load_image
        img = instance._load_image(1) # Second image is png
        assert img.shape == (32, 32)
        assert img.dtype == np.uint16
        assert img[0,0] == 50 * 257 # Scaled from uint8

    def test_load_image_rgb_to_gray(self, dataset_instance_for_load_image):
        instance, _, _, rgb_png_path = dataset_instance_for_load_image
        instance.config.channels = 1 # Ensure config expects 1 channel
        
        img = instance._load_image(2) # Third image is RGB png
        assert img.shape == (32, 32) # Should be grayscale
        assert img.dtype == np.uint16
        # Check a pixel value (cv2.cvtColor with RGB2GRAY uses specific weights)
        # Original: [10,20,30] (BGR if cv2.imread read it, but PIL might be used)
        # If read by cv2.imread, it's BGR: B=10, G=20, R=30
        # Gray = 0.299*R + 0.587*G + 0.114*B = 0.299*30 + 0.587*20 + 0.114*10 = 8.97 + 11.74 + 1.14 = 21.85
        # Then scaled: 21.85 * 257. Expected ~5615. Let's check it's not one of the original channel values.
        assert img[0,0] != 10*257 and img[0,0] != 20*257 and img[0,0] != 30*257


    def test_load_image_index_out_of_bounds(self, dataset_instance_for_load_image):
        instance, _, _, _ = dataset_instance_for_load_image
        with pytest.raises(IndexError):
            instance._load_image(len(instance.image_paths)) # One past the end

    @patch('tifffile.imread', side_effect=IOError("Tif failed"))
    @patch('cv2.imread', return_value=None) # cv2 fails
    @patch('PIL.Image.open', side_effect=IOError("PIL failed"))
    def test_load_image_all_backends_fail(self, mock_pil_open, mock_cv2_imread, mock_tifffile_imread, dataset_instance_for_load_image):
        instance, tif_path, _, _ = dataset_instance_for_load_image
        # Ensure the first image path (tif_path) is used, which will trigger tifffile.imread
        instance.image_paths = np.array([tif_path])
        
        with pytest.raises(IOError, match="Failed to load image .* with OpenCV and PIL: PIL failed"): # Match PIL error as it's the last fallback
            instance._load_image(0)
        mock_tifffile_imread.assert_called_once_with(str(tif_path))
        # cv2.imread would be called if not tif, but here tifffile is first for .tif
        # If we change to a .png path, then cv2 and PIL would be tried.
        
        # Test with a non-TIF file to ensure cv2 and PIL are tried
        png_path_str = str(dataset_instance_for_load_image[2]) # png_path
        instance.image_paths = np.array([Path(png_path_str)]) # Use a png path
        
        # Reset mocks for the new call
        mock_tifffile_imread.reset_mock()
        mock_cv2_imread.reset_mock()
        mock_pil_open.reset_mock()

        with pytest.raises(IOError, match="Failed to load image .* with OpenCV and PIL: PIL failed"):
            instance._load_image(0)
        
        mock_cv2_imread.assert_called_once_with(png_path_str, cv2.IMREAD_UNCHANGED)
        mock_pil_open.assert_called_once_with(Path(png_path_str))


class TestCARSThyroidDatasetPreprocessImage:
    @pytest.fixture
    def instance_for_preprocess(self, base_dataset_config):
         with patch.object(CARSThyroidDataset, '_load_split_data'):
            instance = CARSThyroidDataset(config=base_dataset_config, mode='train')
         return instance

    def test_preprocess_image_resize_normalize_tensor(self, instance_for_preprocess):
        instance_for_preprocess.config.img_size = 64 # Target size
        instance_for_preprocess.config.channels = 1
        
        # Input image (different size, uint16)
        input_img_np = np.arange(32*32, dtype=np.uint16).reshape((32,32)) 
        input_img_np[0,0] = 0
        input_img_np[0,1] = 65535 # Max uint16 value

        tensor_img = instance_for_preprocess._preprocess_image(input_img_np)

        assert isinstance(tensor_img, torch.Tensor)
        assert tensor_img.shape == (1, 64, 64) # C, H, W
        assert tensor_img.dtype == torch.float32
        
        # Check normalization (approximate due to resize interpolation)
        # Original min 0 should be 0.0, original max 65535 should be 1.0
        # This is harder to check precisely after resize.
        # Let's check on an image that doesn't require resize.
        instance_for_preprocess.config.img_size = 2 # Match input_img_np_no_resize dimensions
        input_img_np_no_resize = np.array([[0, 65535],[32767, 10000]], dtype=np.uint16)
        tensor_no_resize = instance_for_preprocess._preprocess_image(input_img_np_no_resize)
        
        assert tensor_no_resize.shape == (1, 2, 2)
        assert torch.isclose(tensor_no_resize[0,0,0], torch.tensor(0.0))
        assert torch.isclose(tensor_no_resize[0,0,1], torch.tensor(1.0))
        assert torch.isclose(tensor_no_resize[0,1,0], torch.tensor(32767.0/65535.0))


class TestCARSThyroidDatasetGetItemLen:
    @pytest.fixture
    def populated_dataset(self, base_dataset_config, create_dummy_image_files):
        # This will run _load_split_data. We need to ensure splits are generated or loaded.
        # For simplicity, let's mock _load_split_data to set up image_paths and labels.
        with patch.object(CARSThyroidDataset, '_load_split_data') as mock_load:
            dataset = CARSThyroidDataset(config=base_dataset_config, mode='train')
        
        # Manually populate after __init__'s mock_load_split_data call
        all_paths = create_dummy_image_files["normal"] + create_dummy_image_files["cancerous"]
        dataset.image_paths = np.array(all_paths[:3]) # Take 3 images for this test split
        dataset.labels = np.array([0,1,0])
        dataset.indices = np.arange(len(dataset.image_paths))
        
        # Mock internal load and preprocess for __getitem__
        dataset._load_image = MagicMock(return_value=np.zeros((32,32), dtype=np.uint16))
        dataset._preprocess_image = MagicMock(return_value=torch.zeros((1,32,32), dtype=torch.float32))
        return dataset

    def test_len(self, populated_dataset):
        assert len(populated_dataset) == 3

    def test_getitem_basic(self, populated_dataset):
        img_tensor, label_tensor = populated_dataset[0]
        
        populated_dataset._load_image.assert_called_with(0)
        # _preprocess_image is called with the result of _load_image
        populated_dataset._preprocess_image.assert_called_with(populated_dataset._load_image.return_value)
        
        assert torch.is_tensor(img_tensor)
        assert img_tensor.shape == (1,32,32)
        assert torch.is_tensor(label_tensor)
        assert label_tensor.item() == 0 # First label was 0
        assert label_tensor.dtype == torch.long

    def test_getitem_with_transform(self, populated_dataset):
        mock_transform_func = MagicMock(return_value=torch.ones((1,32,32))) # Transform returns ones
        populated_dataset.transform = mock_transform_func
        
        img_tensor, _ = populated_dataset[1]
        
        mock_transform_func.assert_called_once_with(populated_dataset._preprocess_image.return_value)
        assert torch.all(img_tensor == 1.0) # Check if transform was applied

    def test_getitem_index_out_of_bounds(self, populated_dataset):
        with pytest.raises(IndexError):
            _ = populated_dataset[len(populated_dataset)]


# --- create_data_loaders Tests ---
@patch('src.data.dataset.CARSThyroidDataset')
def test_create_data_loaders(MockCARSThyroidDataset, base_dataset_config, base_training_config):
    # Mock the CARSThyroidDataset instances
    mock_train_dataset = MagicMock(spec=CARSThyroidDataset)
    mock_train_dataset.__len__.return_value = 10
    
    mock_val_dataset = MagicMock(spec=CARSThyroidDataset)
    mock_val_dataset.__len__.return_value = 5
    
    mock_test_dataset = MagicMock(spec=CARSThyroidDataset)
    mock_test_dataset.__len__.return_value = 0 # Empty test set

    MockCARSThyroidDataset.side_effect = [mock_train_dataset, mock_val_dataset, mock_test_dataset]

    mock_transform_train = MagicMock()
    mock_transform_val = MagicMock()

    dataloaders = create_data_loaders(
        dataset_config=base_dataset_config,
        training_config=base_training_config,
        transform_train=mock_transform_train,
        transform_val=mock_transform_val
    )

    assert MockCARSThyroidDataset.call_count == 3
    calls = [
        call(config=base_dataset_config, mode='train', transform=mock_transform_train),
        call(config=base_dataset_config, mode='val', transform=mock_transform_val),
        call(config=base_dataset_config, mode='test', transform=mock_transform_val) # val transform for test
    ]
    MockCARSThyroidDataset.assert_has_calls(calls)

    assert 'train' in dataloaders
    assert 'val' in dataloaders
    assert 'test' in dataloaders

    assert dataloaders['train'].batch_size == base_training_config.batch_size
    assert isinstance(dataloaders['train'].sampler, torch.utils.data.RandomSampler) # Check sampler for shuffle
    assert dataloaders['train'].drop_last is True # Since len > 0

    assert isinstance(dataloaders['val'].sampler, torch.utils.data.SequentialSampler) # Check sampler for no shuffle
    assert dataloaders['val'].drop_last is False

    assert dataloaders['test'].dataset == mock_test_dataset
    assert len(dataloaders['test'].dataset) == 0
    assert dataloaders['test'].drop_last is False # Even if train mode, len is 0

    # Check console output for empty dataset warning
    # src.data.dataset.console.print.assert_any_call("[yellow]Warning: Dataset for split 'test' is empty. DataLoader will also be empty.[/yellow]")


# TODO: Add more detailed tests for _load_split_data logic (various file scenarios, k-fold, generation)
# This is the most complex part and needs careful mocking of file system and json loads.

# Example of how to start testing _load_split_data (very simplified)
class TestCARSThyroidDatasetLoadSplitData:

    @patch('src.data.dataset.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(CARSThyroidDataset, '_get_all_image_metadata')
    def test_load_split_data_specific_file_success(self, mock_get_metadata, mock_file_open, mock_path_exists, base_dataset_config, tmp_path):
        base_dataset_config.split_file = str(tmp_path / "my_split.json")
        base_dataset_config.use_kfold = False # Ensure not k-fold path

        # Mock return for _get_all_image_metadata
        num_total_images = 10
        mock_paths = np.array([Path(f"img_{i}.tif") for i in range(num_total_images)])
        mock_labels = np.array([i % 2 for i in range(num_total_images)])
        mock_get_metadata.return_value = (mock_paths, mock_labels, np.array([]))

        # Mock file content
        split_content = {"train": [0, 1, 2], "val": [3, 4], "test": [5,6]}
        mock_file_open.return_value.read.return_value = json.dumps(split_content)
        mock_path_exists.return_value = True # Split file exists

        dataset = CARSThyroidDataset(config=base_dataset_config, mode='train') # __init__ calls _load_split_data

        mock_get_metadata.assert_called_once()
        mock_file_open.assert_called_once_with(Path(base_dataset_config.split_file), 'r')
        
        assert len(dataset.image_paths) == len(split_content['train'])
        assert np.array_equal(dataset.image_paths, mock_paths[split_content['train']])
        assert np.array_equal(dataset.labels, mock_labels[split_content['train']])
        assert np.array_equal(dataset.indices, np.arange(len(split_content['train'])))
        assert dataset.current_split_file == Path(base_dataset_config.split_file)

    # Add tests for file not found, json errors, mode not in file, index errors etc. for _load_split_data
    # Add tests for k-fold logic in _load_split_data
    # Add tests for non-kfold, non-specific-file (split generation/loading from split_info.json)