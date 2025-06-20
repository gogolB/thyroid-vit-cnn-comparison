import timm
import torch.nn as nn
from src.models.base import ModelBase

class VisionTransformer(ModelBase):
    """
    Unified Vision Transformer (ViT) implementation.
    Relies on timm for model creation.
    """
    def __init__(self, config):
        """
        Initializes the VisionTransformer model.

        Args:
            config: A configuration object containing model parameters like
                    name (variant), pretrained, num_classes, img_size, patch_size.
        """
        super().__init__(config) # Pass config to ModelBase
        self.variant = config.name  # e.g., 'vit_base_patch16_224'
        self._build_model()

    def _build_model(self):
        """
        Builds the ViT model based on the variant and parameters in the config.
        Uses the timm library.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2)
        img_size = self.config.get('img_size', 224)
        patch_size = self.config.get('patch_size', 16)
        # Prioritize in_chans from extra_params, then config.channels, then default to 3 (or 1 for CARS)
        in_chans_extra = self.config.extra_params.get('in_chans') if hasattr(self.config, 'extra_params') else None
        in_chans_dataset = self.config.get('channels') # From DatasetConfig part if merged
        # Default to 1 channel if not specified
        in_chans = in_chans_extra if in_chans_extra is not None else (in_chans_dataset if in_chans_dataset is not None else 1)
        
        # Remove the hardcoded img_size override
        
        # Construct the model name for timm.
        # For standard ViT models, append patch size and image size.
        # e.g., 'vit_tiny' -> 'vit_tiny_patch16_224'
        model_name_parts = [self.variant]
        if "patch" not in self.variant: # Avoid double "patch" if already in name
            model_name_parts.append(f"patch{patch_size}")
        if str(img_size) not in self.variant: # Avoid double img_size if already in name
             model_name_parts.append(str(img_size))
        
        model_name = "_".join(model_name_parts)
        
        # Handle common timm naming variations for vit_tiny, vit_small, vit_base
        if self.variant == 'vit_tiny':
            model_name = f'vit_tiny_patch{patch_size}_{img_size}'
        elif self.variant == 'vit_small':
            model_name = f'vit_small_patch{patch_size}_{img_size}'
        elif self.variant == 'vit_base':
            model_name = f'vit_base_patch{patch_size}_{img_size}'
        # For other ViT variants from config.name, assume they are already timm-compatible
        # or the generic construction above works.

        print(f"Attempting to create ViT model with timm name: {model_name}")

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_chans, # Pass in_chans
                img_size=img_size
            )
        except Exception as e:
            print(f"Initial timm.create_model failed for {model_name} with img_size={img_size}, in_chans={in_chans}: {e}. Retrying common variations.")
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    in_chans=in_chans
                    # img_size might be implied by model_name
                )
                print(f"Successfully created {model_name} on retry (with in_chans, without explicit img_size).")
            except Exception as e_retry:
                print(f"Retry for {model_name} failed. Trying original variant name {self.variant} with in_chans.")
                try:
                    self.model = timm.create_model(
                        self.variant,
                        pretrained=pretrained,
                        num_classes=num_classes,
                        in_chans=in_chans,
                        img_size=img_size
                    )
                    print(f"Successfully created {self.variant} using original config name (with in_chans).")
                except Exception as e_original_variant:
                    raise RuntimeError(f"Failed to create ViT model. Tried '{model_name}' and '{self.variant}' with in_chans={in_chans}. Last error: {e_original_variant}")