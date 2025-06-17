import os
import subprocess
import threading

# Define paths to model configuration directories
CNN_MODEL_CONFIG_DIR = "configs/model/cnn/"
VIT_MODEL_CONFIG_DIR = "configs/model/vit/"

# Files to exclude from model discovery
CNN_EXCLUDE_FILES = ["base.yaml", "base_cnn.yaml"] # Added base_cnn.yaml
VIT_EXCLUDE_FILES = ["__init__.yaml", "base_transformer.yaml"]

def discover_models():
    """Discovers CNN and ViT models from their configuration directories."""
    models = []

    # Discover CNN models
    if os.path.exists(CNN_MODEL_CONFIG_DIR):
        for f_name in sorted(os.listdir(CNN_MODEL_CONFIG_DIR)): # sorted for consistent order
            if f_name.endswith(".yaml") and f_name not in CNN_EXCLUDE_FILES:
                model_name = f_name[:-5]  # Remove .yaml
                models.append({"name": model_name, "type": "cnn"})
    else:
        print(f"Warning: CNN model config directory not found at {CNN_MODEL_CONFIG_DIR}")

    # Discover ViT models
    if os.path.exists(VIT_MODEL_CONFIG_DIR):
        for f_name in sorted(os.listdir(VIT_MODEL_CONFIG_DIR)): # sorted for consistent order
            if f_name.endswith(".yaml") and f_name not in VIT_EXCLUDE_FILES:
                model_name = f_name[:-5]  # Remove .yaml
                models.append({"name": model_name, "type": "vit"})
    else:
        print(f"Warning: ViT model config directory not found at {VIT_MODEL_CONFIG_DIR}")
    
    return models

def run_quick_test(model_name, model_type):
    """Constructs and runs the quick test command for a given model."""
    print(f"\n--- Preparing test for model: {model_name} (type: {model_type}) ---")

    # Determine the correct training configuration based on model type
    # As per instructions, 'training=vit' is assumed valid for ViT models.
    # 'training=cnn' for CNN models.
    training_config = "cnn" if model_type == "cnn" else "transformer"

    command = [
        "python", "-m", "src.experiment.manager",
        f"model={model_type}/{model_name}",
        "dataset=default",
        "trainer=default",
        f"training={training_config}",
        "augmentation=no_aug",
        "kfold.is_primary_kfold_experiment=true",
        "kfold.num_folds=2",
        "trainer.max_epochs=1",
        f"hydra.job.name=quick_kfold2_epoch1_{model_type}_{model_name}", # Corrected hydra job name override
        "params.use_wandb=false"
    ]

    print(f"Executing command: {' '.join(command)}")

    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1  # Line buffered
        )
        
        # Function to capture and print stream output
        def capture_stream(stream, stream_name):
            for line in stream:
                print(f"[{stream_name}] {line}", end='')
                yield line
                
        # Create threads to capture both stdout and stderr
        stdout_lines = []
        stderr_lines = []
        
        stdout_thread = threading.Thread(
            target=lambda: [line for line in capture_stream(process.stdout, "stdout")]
        )
        stderr_thread = threading.Thread(
            target=lambda: [line for line in capture_stream(process.stderr, "stderr")]
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        
        if return_code == 0:
            print(f"--- Successfully tested model: {model_name} (type: {model_type}) ---")
            return True
        else:
            print(f"!!! Error testing model: {model_name} (type: {model_type}) !!!")
            print(f"Return code: {return_code}")
            return False
            
    except FileNotFoundError:
        print("!!! Error: python interpreter or src.experiment.manager not found. Ensure your environment is set up correctly. !!!")
        return False
    except Exception as e:
        print(f"!!! An unexpected error occurred while testing model: {model_name} (type: {model_type}) !!!")
        print(str(e))
        return False

def main():
    """Main function to discover models and run quick tests."""
    models_to_test = discover_models()

    if not models_to_test:
        print("No models found to test. Exiting.")
        return

    print(f"Found {len(models_to_test)} models to test: {[m['type']+'/'+m['name'] for m in models_to_test]}")

    successful_tests = 0
    failed_tests = 0

    for model_info in models_to_test:
        if run_quick_test(model_info["name"], model_info["type"]):
            successful_tests += 1
        else:
            failed_tests += 1
    
    print("\n--- All model tests completed. ---")
    print(f"Summary: {successful_tests} successful, {failed_tests} failed.")

if __name__ == "__main__":
    main()