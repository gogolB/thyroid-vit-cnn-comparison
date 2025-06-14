#!/usr/bin/env python3
"""
Model loading utility that works with the actual codebase structure.
Properly loads CNN and ViT models from checkpoints.
File: scripts/model_loader_utils.py
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Dict, Any
from omegaconf import OmegaConf, DictConfig
import warnings
from rich.console import Console

console = Console()


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = 'cpu',
    verbose: bool = True
) -> Optional[nn.Module]:
    """
    Load a model from checkpoint using the Lightning modules.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        verbose: Whether to print loading messages
        
    Returns:
        Loaded model in eval mode or None if loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        if verbose:
            console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        return None
    
    if verbose:
        console.print(f"[cyan]Loading model from {checkpoint_path}...[/cyan]")
    
    try:
        # Determine model type from filename
        filename = checkpoint_path.stem.lower()
        
        # Import the appropriate Lightning module
        if any(vit_type in filename for vit_type in ['vit', 'deit', 'swin']):
            # Load as Vision Transformer
            from src.training.train_vit import ThyroidViTModule
            
            # Load with weights_only=False for PyTorch 2.6
            model_module = ThyroidViTModule.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                weights_only=False,
                strict=False
            )
            
            if verbose:
                console.print(f"[green]✓ Successfully loaded ViT model[/green]")
        else:
            # Load as CNN
            from src.training.train_cnn import ThyroidCNNModule
            
            model_module = ThyroidCNNModule.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                weights_only=False,
                strict=False
            )
            
            if verbose:
                console.print(f"[green]✓ Successfully loaded CNN model[/green]")
        
        # Set to eval mode
        model_module.eval()
        
        # Disable gradient computation
        for param in model_module.parameters():
            param.requires_grad = False
        
        # Move to device
        model_module = model_module.to(device)
        
        return model_module
        
    except Exception as e:
        if verbose:
            console.print(f"[red]Error loading model: {e}[/red]")
            
            # Try a more direct approach
            console.print("[yellow]Attempting direct state_dict loading...[/yellow]")
            
        try:
            # Load checkpoint directly
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Extract model info
            if 'hyper_parameters' in checkpoint:
                hp = checkpoint['hyper_parameters']
                model_name = None
                
                if 'model' in hp and isinstance(hp['model'], dict):
                    model_name = hp['model'].get('name', '')
                elif 'config' in hp and isinstance(hp['config'], dict):
                    if 'model' in hp['config'] and isinstance(hp['config']['model'], dict):
                        model_name = hp['config']['model'].get('name', '')
                
                if not model_name:
                    model_name = filename.split('-')[0]
                
                # Create the model based on type
                if 'swin' in model_name:
                    # For Swin, we need to handle the specific architecture
                    from src.models.vit import get_vit_model
                    
                    # Extract the actual Swin configuration from the checkpoint
                    state_dict = checkpoint.get('state_dict', {})
                    
                    # Infer model configuration from state dict
                    # This is a bit hacky but necessary given the errors
                    embed_dim = None
                    for key in state_dict.keys():
                        if 'patch_embed.proj.weight' in key:
                            # Shape is [embed_dim, in_chans, patch_size, patch_size]
                            embed_dim = state_dict[key].shape[0]
                            break
                    
                    if embed_dim == 96:
                        model = get_vit_model('swin_tiny', in_chans=1, num_classes=2, pretrained=False)
                    elif embed_dim == 192:
                        model = get_vit_model('swin_small', in_chans=1, num_classes=2, pretrained=False)
                    else:
                        # Default to tiny
                        model = get_vit_model('swin_tiny', in_chans=1, num_classes=2, pretrained=False)
                    
                    # Load state dict with strict=False
                    if 'state_dict' in checkpoint:
                        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                        model.load_state_dict(state_dict, strict=False)
                    
                    model.eval()
                    model = model.to(device)
                    
                    if verbose:
                        console.print(f"[green]✓ Successfully loaded Swin model via state_dict[/green]")
                    
                    return model
                    
        except Exception as e2:
            if verbose:
                console.print(f"[red]Direct loading also failed: {e2}[/red]")
        
        return None


def get_model_output(model: Union[nn.Module, 'pl.LightningModule'], x: torch.Tensor) -> torch.Tensor:
    """
    Get model output, handling different output formats and module types.
    
    Args:
        model: The model (either raw nn.Module or Lightning module)
        x: Input tensor
        
    Returns:
        Logits tensor of shape [batch_size, num_classes]
    """
    # Check if it's a Lightning module
    if hasattr(model, 'model'):
        # It's a Lightning module, get the underlying model
        actual_model = model.model
    else:
        actual_model = model
    
    # Ensure eval mode
    model.eval()
    if hasattr(actual_model, 'eval'):
        actual_model.eval()
    
    with torch.no_grad():
        # For Lightning modules, use forward method
        if hasattr(model, 'forward') and hasattr(model, 'model'):
            output = model(x)
        else:
            output = actual_model(x)
    
    # Handle different output formats
    if isinstance(output, dict):
        # CNN models from MedicalResNet return dict with 'logits'
        if 'logits' in output:
            return output['logits']
        elif 'output' in output:
            return output['output']
        else:
            # Some models might return other dict formats
            # Try to find anything that looks like logits
            for key, val in output.items():
                if isinstance(val, torch.Tensor) and val.dim() == 2:
                    return val
            raise ValueError(f"Unknown dict output format: {output.keys()}")
    elif isinstance(output, (list, tuple)):
        # Some models return tuple (logits, features, ...)
        return output[0]
    else:
        # Direct tensor output
        return output


def debug_model_output(model: nn.Module, sample_input: torch.Tensor, model_name: str = "unknown"):
    """Debug helper to understand model output format."""
    model.eval()
    console.print(f"\n[cyan]Debug info for {model_name}:[/cyan]")
    
    # Check if it's a Lightning module
    if hasattr(model, 'model'):
        console.print("  Model type: Lightning module")
        console.print(f"  Underlying model: {type(model.model)}")
    else:
        console.print(f"  Model type: {type(model)}")
    
    with torch.no_grad():
        try:
            output = model(sample_input)
        except Exception as e:
            console.print(f"[red]  Forward pass error: {e}[/red]")
            return None
    
    console.print(f"  Output type: {type(output)}")
    
    if isinstance(output, dict):
        console.print(f"  Dict keys: {output.keys()}")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                console.print(f"    {key}: shape={val.shape}, range=[{val.min():.3f}, {val.max():.3f}]")
    elif isinstance(output, torch.Tensor):
        console.print(f"  Tensor shape: {output.shape}")
        console.print(f"  Tensor range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check if outputs are reasonable
        if output.shape[1] == 2:  # Binary classification
            probs = torch.softmax(output, dim=1)
            console.print(f"  Softmax probs sample: {probs[0].tolist()}")
            console.print(f"  Predicted class: {torch.argmax(output, dim=1).tolist()}")
    
    return output