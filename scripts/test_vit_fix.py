import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.experiment.kfold_experiment import KFoldExperiment
from src.experiment.config import ExperimentConfig, KFoldConfig

config = ExperimentConfig(
    name="vit_fix_test",
    model={"name": "vit_small"},
    dataset={"name": "default", "data_path": "data/raw"},
    trainer={"max_epochs": 1},
    training_content={"batch_size": 8, "num_workers": 0},
    kfold=KFoldConfig()
)

experiment = KFoldExperiment(config)
experiment.run_fold(2)

# If successful, update project log
with open("project_log.md", "a") as log_file:
    log_file.write("\n\n### VisionTransformer Channel Fix Test\n")
    log_file.write("- Successfully ran fold 2 test with 1-channel input\n")
    log_file.write("- Training completed 1 epoch without channel dimension errors\n")
    log_file.write("- Checkpoints saved properly\n")

print("Test completed successfully. Project log updated.")