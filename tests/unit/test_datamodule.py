import pytest
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace
from torch.utils.data import DataLoader as TorchDataLoader # Actual class

from src.data.datamodule import ThyroidDataModule

@pytest.fixture
def mock_dataset_config_data():
    return {
        "img_size": 224,
        "channels": 3,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "apply_augmentations": True,
        "data_dir": "fake/data",
        "split_dir": "fake/splits",
        "fold": 1,
        "use_quality_preprocessing": False,
        "quality_config_path": "fake/quality.yaml",
        "use_vit_transforms": False,
        "model_name": "resnet50"
    }

@pytest.fixture
def dataset_config(mock_dataset_config_data):
    return SimpleNamespace(**mock_dataset_config_data)

@pytest.fixture
def training_config_data():
    return {
        "batch_size": 32,
        "num_workers": 4
    }

@pytest.fixture
def training_config(training_config_data):
    return SimpleNamespace(**training_config_data)

@pytest.fixture
def datamodule(dataset_config, training_config):
    with patch.object(ThyroidDataModule, 'save_hyperparameters') as mock_save_hp:
        dm = ThyroidDataModule(dataset_config=dataset_config, training_config=training_config)
        mock_save_hp.assert_called_once_with(logger=False)
        return dm

class TestThyroidDataModule:

    def test_init(self, datamodule, dataset_config, training_config):
        assert datamodule.dataset_config == dataset_config
        assert datamodule.training_config == training_config
        assert datamodule.train_transforms is None
        assert datamodule.val_transforms is None
        assert datamodule.test_transforms is None
        assert datamodule.train_dataset is None
        assert datamodule.val_dataset is None
        assert datamodule.test_dataset is None
        assert datamodule.predict_dataset is None

    def test_prepare_data(self, datamodule):
        datamodule.prepare_data() # Should not raise errors

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_stage_fit(self, mock_get_transforms, mock_cars_dataset, datamodule, dataset_config):
        mock_transform_train = MagicMock(name="train_transform")
        mock_transform_val = MagicMock(name="val_transform")
        mock_transform_test = MagicMock(name="test_transform")
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, mock_transform_test]

        mock_train_ds = MagicMock(name="train_ds")
        mock_train_ds.image_paths = MagicMock(); mock_train_ds.image_paths.size = 10
        mock_val_ds = MagicMock(name="val_ds")
        mock_val_ds.image_paths = MagicMock(); mock_val_ds.image_paths.size = 5
        mock_cars_dataset.side_effect = [mock_train_ds, mock_val_ds]

        datamodule.setup(stage='fit')

        expected_get_transforms_calls = [
            call(dataset_config, mode='train'),
            call(dataset_config, mode='val'),
            call(dataset_config, mode='test'),
        ]
        mock_get_transforms.assert_has_calls(expected_get_transforms_calls)
        assert datamodule.train_transforms == mock_transform_train
        assert datamodule.val_transforms == mock_transform_val
        assert datamodule.test_transforms == mock_transform_test

        expected_cars_dataset_calls = [
            call(config=dataset_config, mode='train', transform=mock_transform_train),
            call(config=dataset_config, mode='val', transform=mock_transform_val),
        ]
        mock_cars_dataset.assert_has_calls(expected_cars_dataset_calls)
        assert datamodule.train_dataset == mock_train_ds
        assert datamodule.val_dataset == mock_val_ds
        assert datamodule.test_dataset is None
        assert datamodule.predict_dataset is None

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_stage_test(self, mock_get_transforms, mock_cars_dataset, datamodule, dataset_config):
        mock_transform_train, mock_transform_val, mock_transform_test = MagicMock(), MagicMock(), MagicMock()
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, mock_transform_test]
        mock_test_ds = MagicMock(name="test_ds")
        mock_cars_dataset.return_value = mock_test_ds

        datamodule.setup(stage='test')

        mock_get_transforms.assert_any_call(dataset_config, mode='test')
        assert datamodule.test_transforms == mock_transform_test
        mock_cars_dataset.assert_called_once_with(config=dataset_config, mode='test', transform=mock_transform_test)
        assert datamodule.test_dataset == mock_test_ds
        assert datamodule.train_dataset is None
        assert datamodule.val_dataset is None
        assert datamodule.predict_dataset is None

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_stage_predict(self, mock_get_transforms, mock_cars_dataset, datamodule, dataset_config):
        mock_transform_train, mock_transform_val, mock_transform_test = MagicMock(), MagicMock(), MagicMock()
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, mock_transform_test]
        mock_predict_ds = MagicMock(name="predict_ds")
        mock_cars_dataset.return_value = mock_predict_ds

        datamodule.setup(stage='predict')
        
        mock_get_transforms.assert_any_call(dataset_config, mode='test')
        assert datamodule.test_transforms == mock_transform_test
        mock_cars_dataset.assert_called_once_with(config=dataset_config, mode='test', transform=mock_transform_test)
        assert datamodule.predict_dataset == mock_predict_ds
        assert datamodule.train_dataset is None
        assert datamodule.val_dataset is None
        assert datamodule.test_dataset is None

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_stage_none(self, mock_get_transforms, mock_cars_dataset, datamodule, dataset_config):
        mock_transform_train, mock_transform_val, mock_transform_test = MagicMock(), MagicMock(), MagicMock()
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, mock_transform_test]

        mock_train_ds, mock_val_ds, mock_test_ds, mock_predict_ds = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        for ds_mock in [mock_train_ds, mock_val_ds, mock_test_ds, mock_predict_ds]:
            ds_mock.image_paths = MagicMock(); ds_mock.image_paths.size = 1
        mock_cars_dataset.side_effect = [mock_train_ds, mock_val_ds, mock_test_ds, mock_predict_ds]

        datamodule.setup(stage=None)

        assert datamodule.train_transforms == mock_transform_train
        assert datamodule.val_transforms == mock_transform_val
        assert datamodule.test_transforms == mock_transform_test
        assert datamodule.train_dataset == mock_train_ds
        assert datamodule.val_dataset == mock_val_ds
        assert datamodule.test_dataset == mock_test_ds
        assert datamodule.predict_dataset == mock_predict_ds
        expected_cars_calls = [
            call(config=dataset_config, mode='train', transform=mock_transform_train),
            call(config=dataset_config, mode='val', transform=mock_transform_val),
            call(config=dataset_config, mode='test', transform=mock_transform_test),
            call(config=dataset_config, mode='test', transform=mock_transform_test)
        ]
        mock_cars_dataset.assert_has_calls(expected_cars_calls)

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_empty_train_dataset_raises_error(self, mock_get_transforms, mock_cars_dataset, datamodule):
        mock_transform_train, mock_transform_val, _ = MagicMock(), MagicMock(), MagicMock()
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, _]
        mock_empty_train_ds = MagicMock(name="empty_train_ds")
        mock_empty_train_ds.image_paths = MagicMock(); mock_empty_train_ds.image_paths.size = 0
        mock_val_ds = MagicMock(name="val_ds"); mock_val_ds.image_paths = MagicMock(); mock_val_ds.image_paths.size = 1
        mock_cars_dataset.side_effect = [mock_empty_train_ds, mock_val_ds]

        with pytest.raises(ValueError, match="Training dataset is empty"):
            datamodule.setup(stage='fit')

    @patch('src.data.datamodule.CARSThyroidDataset')
    @patch('src.data.datamodule.get_transforms')
    def test_setup_empty_val_dataset_raises_error(self, mock_get_transforms, mock_cars_dataset, datamodule):
        mock_transform_train, mock_transform_val, _ = MagicMock(), MagicMock(), MagicMock()
        mock_get_transforms.side_effect = [mock_transform_train, mock_transform_val, _]
        mock_train_ds = MagicMock(name="train_ds"); mock_train_ds.image_paths = MagicMock(); mock_train_ds.image_paths.size = 1
        mock_empty_val_ds = MagicMock(name="empty_val_ds")
        mock_empty_val_ds.image_paths = MagicMock(); mock_empty_val_ds.image_paths.size = 0
        mock_cars_dataset.side_effect = [mock_train_ds, mock_empty_val_ds]

        with pytest.raises(ValueError, match="Validation dataset is empty"):
            datamodule.setup(stage='fit')

    @patch('src.data.datamodule.DataLoader')
    def test_train_dataloader_success(self, mock_torch_dataloader, datamodule, training_config):
        datamodule.train_dataset = MagicMock(name="train_dataset_instance")
        datamodule._has_setup_fit = {'fit'}; datamodule._trainer = None
        loader_instance = MagicMock(name="dataloader_instance")
        mock_torch_dataloader.return_value = loader_instance
        
        result = datamodule.train_dataloader()

        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            datamodule.train_dataset,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            shuffle=True, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    def test_train_dataloader_not_setup_runtime_error(self, datamodule):
        datamodule.train_dataset = None; datamodule._has_setup_fit = set(); datamodule._trainer = None
        with pytest.raises(RuntimeError, match="Train dataset not setup"):
            datamodule.train_dataloader()

    @patch('src.data.datamodule.ThyroidDataModule.setup')
    @patch('src.data.datamodule.DataLoader')
    def test_train_dataloader_calls_setup_if_needed(self, mock_torch_dataloader, mock_dm_setup, datamodule, training_config):
        datamodule.train_dataset = None; datamodule._has_setup_fit = set(); datamodule._trainer = MagicMock()
        mock_setup_train_dataset = MagicMock(name="setup_train_ds")
        def setup_side_effect(stage):
            if stage == 'fit': datamodule.train_dataset = mock_setup_train_dataset
        mock_dm_setup.side_effect = setup_side_effect
        loader_instance = MagicMock(); mock_torch_dataloader.return_value = loader_instance

        result = datamodule.train_dataloader()

        mock_dm_setup.assert_called_once_with(stage='fit')
        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            mock_setup_train_dataset, batch_size=training_config.batch_size,
            num_workers=training_config.num_workers, shuffle=True, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    @patch('src.data.datamodule.DataLoader')
    def test_val_dataloader_success(self, mock_torch_dataloader, datamodule, training_config):
        datamodule.val_dataset = MagicMock(name="val_dataset_instance")
        datamodule._has_setup_fit = {'fit'}; datamodule._trainer = None
        loader_instance = MagicMock(); mock_torch_dataloader.return_value = loader_instance
        
        result = datamodule.val_dataloader()

        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            datamodule.val_dataset, batch_size=training_config.batch_size,
            num_workers=training_config.num_workers, shuffle=False, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    def test_val_dataloader_not_setup_runtime_error(self, datamodule):
        datamodule.val_dataset = None; datamodule._has_setup_fit = set(); datamodule._trainer = None
        with pytest.raises(RuntimeError, match="Validation dataset not setup"):
            datamodule.val_dataloader()

    @patch('src.data.datamodule.ThyroidDataModule.setup')
    @patch('src.data.datamodule.DataLoader')
    def test_val_dataloader_calls_setup_if_needed(self, mock_torch_dataloader, mock_dm_setup, datamodule, training_config):
        datamodule.val_dataset = None; datamodule._has_setup_fit = set(); datamodule._trainer = MagicMock()
        mock_setup_val_dataset = MagicMock(name="setup_val_ds")
        def setup_side_effect(stage):
            if stage == 'fit': datamodule.val_dataset = mock_setup_val_dataset
        mock_dm_setup.side_effect = setup_side_effect
        loader_instance = MagicMock(); mock_torch_dataloader.return_value = loader_instance

        result = datamodule.val_dataloader()

        mock_dm_setup.assert_called_once_with(stage='fit')
        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            mock_setup_val_dataset, batch_size=training_config.batch_size,
            num_workers=training_config.num_workers, shuffle=False, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    @patch('src.data.datamodule.DataLoader')
    def test_test_dataloader_success(self, mock_torch_dataloader, datamodule, training_config):
        datamodule.test_dataset = MagicMock(name="test_dataset_instance")
        datamodule._has_setup_test = {'test'}; datamodule._trainer = None
        loader_instance = MagicMock(); mock_torch_dataloader.return_value = loader_instance
        
        result = datamodule.test_dataloader()

        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            datamodule.test_dataset, batch_size=training_config.batch_size,
            num_workers=training_config.num_workers, shuffle=False, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    @patch('builtins.print')
    @patch('src.data.datamodule.ThyroidDataModule.setup')
    def test_test_dataloader_dataset_none_after_setup_returns_none_warns(self, mock_dm_setup, mock_print, datamodule):
        datamodule.test_dataset = None; datamodule._has_setup_test = set(); datamodule._trainer = MagicMock()
        def setup_leaves_none(stage):
            if stage == 'test': datamodule.test_dataset = None
        mock_dm_setup.side_effect = setup_leaves_none
            
        result = datamodule.test_dataloader()
        
        assert result is None
        mock_dm_setup.assert_called_once_with(stage='test')
        mock_print.assert_called_with("Warning: Test dataset is empty after attempting setup in test_dataloader. Returning None.")

    @patch('src.data.datamodule.DataLoader')
    def test_predict_dataloader_success(self, mock_torch_dataloader, datamodule, training_config):
        datamodule.predict_dataset = MagicMock(name="predict_dataset_instance")
        datamodule._has_setup_predict = {'predict'}; datamodule._trainer = None
        loader_instance = MagicMock(); mock_torch_dataloader.return_value = loader_instance
        
        result = datamodule.predict_dataloader()

        assert result == loader_instance
        mock_torch_dataloader.assert_called_once_with(
            datamodule.predict_dataset, batch_size=training_config.batch_size,
            num_workers=training_config.num_workers, shuffle=False, pin_memory=True,
            persistent_workers=training_config.num_workers > 0
        )

    @patch('builtins.print')
    @patch('src.data.datamodule.ThyroidDataModule.setup')
    def test_predict_dataloader_dataset_none_after_setup_returns_none_warns(self, mock_dm_setup, mock_print, datamodule):
        datamodule.predict_dataset = None; datamodule._has_setup_predict = set(); datamodule._trainer = MagicMock()
        def setup_leaves_none(stage):
            if stage == 'predict': datamodule.predict_dataset = None
        mock_dm_setup.side_effect = setup_leaves_none
            
        result = datamodule.predict_dataloader()

        assert result is None
        mock_dm_setup.assert_called_once_with(stage='predict')
        mock_print.assert_called_with("Warning: Predict dataset not setup. Call setup('predict') first or ensure trainer calls it. Returning None.")

    @patch('src.data.datamodule.DataLoader')
    def test_dataloader_persistent_workers_false_if_zero_workers(self, mock_torch_dataloader, dataset_config):
        training_config_zero_workers = SimpleNamespace(batch_size=32, num_workers=0)
        with patch.object(ThyroidDataModule, 'save_hyperparameters'):
            dm_zero_workers = ThyroidDataModule(dataset_config=dataset_config, training_config=training_config_zero_workers)
        
        dm_zero_workers.train_dataset = MagicMock(name="train_ds_instance")
        dm_zero_workers._has_setup_fit = {'fit'}; dm_zero_workers._trainer = None
        mock_torch_dataloader.return_value = MagicMock()
        dm_zero_workers.train_dataloader()

        mock_torch_dataloader.assert_called_once_with(
            dm_zero_workers.train_dataset, batch_size=training_config_zero_workers.batch_size,
            num_workers=training_config_zero_workers.num_workers, shuffle=True, pin_memory=True,
            persistent_workers=False # Key assertion
        )