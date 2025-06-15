import timm
import torch.nn as nn
from src.models.base import ModelBase

class DeiT(ModelBase):
    """
    Unified Data-efficient Image Transformer (DeiT) implementation.
    Relies on timm for model creation.
    """
    def __init__(self, config):
        """
        Initializes the DeiT model.

        Args:
            config: A configuration object containing model parameters.
        """
        super().__init__(config)
        self.variant = config.name  # e.g., 'deit_base_patch16_224'
        self._build_model()

    def _build_model(self):
        """
        Builds the DeiT model based on the variant and parameters in the config.
        Uses the timm library.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2)
        img_size = self.config.get('img_size', 224) # DeiT models in timm often take img_size
        in_chans = self.config.get('in_channels', self.config.get('channels', 1)) # Get in_channels from model or dataset config

        # Mapping from registered short names to full timm model names
        timm_model_name_map = {
            'deit_tiny': 'deit_tiny_patch16_224',
            'deit_small': 'deit_small_patch16_224',
            'deit_base': 'deit_base_patch16_224',
            # Add other deit variants here if registered with short names
        }
        
        # Use the mapped name if available, otherwise use the variant directly
        # This allows for registering with full timm names as well.
        model_name_for_timm = timm_model_name_map.get(self.variant, self.variant)

        try:
            self.model = timm.create_model(
                model_name_for_timm,
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=img_size,
                in_chans=in_chans
            )
        except TypeError as e:
            # Some timm models might not accept img_size or in_chans directly in create_model
            # if the model variant name already implies them or they are fixed.
            # We'll try common fallbacks.
            if 'img_size' in str(e) and 'in_chans' in str(e):
                print(f"Warning: img_size and in_chans caused TypeError for {model_name_for_timm}. Retrying without them.")
                self.model = timm.create_model(model_name_for_timm, pretrained=pretrained, num_classes=num_classes)
            elif 'img_size' in str(e):
                print(f"Warning: img_size caused TypeError for {model_name_for_timm}. Retrying without explicit img_size.")
                self.model = timm.create_model(model_name_for_timm, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
            elif 'in_chans' in str(e):
                 print(f"Warning: in_chans caused TypeError for {model_name_for_timm}. Retrying without explicit in_chans.")
                 self.model = timm.create_model(model_name_for_timm, pretrained=pretrained, num_classes=num_classes, img_size=img_size)
            else:
                # This 'else' means the TypeError was not one of the recognized patterns (img_size/in_chans related).
                # We re-raise the original TypeError.
                raise RuntimeError(f"Failed to create DeiT model {model_name_for_timm} due to unhandled TypeError: {e}")
        except Exception as e: # Catch-all for other timm.create_model errors not TypeError
            raise RuntimeError(f"Failed to create DeiT model {model_name_for_timm} using timm (general error): {e}")