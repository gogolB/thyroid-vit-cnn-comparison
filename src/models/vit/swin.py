import timm
import torch.nn as nn
from src.models.base import ModelBase 


class SwinTransformer(ModelBase):
    """
    Unified Swin Transformer implementation.
    Relies on timm for model creation.
    """
    def __init__(self, config):
        """
        Initializes the SwinTransformer model.

        Args:
            config: A configuration object containing model parameters.
        """
        super().__init__(config)
        self.variant = config.name  # e.g., 'swin_tiny_patch4_window7_224'
        # Assuming super().__init__(config) makes config available as self.config
        self._build_model()

    def _build_model(self):
        """
        Builds the Swin Transformer model based on the variant and parameters in the config.
        Uses the timm library.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2)
        img_size = self.config.get('img_size', 224)
        # Prioritize in_chans from extra_params, then config.channels, then default to 1
        in_chans_extra = self.config.extra_params.get('in_chans') if hasattr(self.config, 'extra_params') else None
        in_chans_dataset = self.config.get('channels') # From DatasetConfig part if merged
        in_chans = in_chans_extra if in_chans_extra is not None else (in_chans_dataset if in_chans_dataset is not None else 1)

        # Mapping from registered short names to full timm model names
        timm_model_name_map = {
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224',
            # Add other swin variants here if registered with short names
            # e.g. 'swin_large': 'swin_large_patch4_window7_224' or 'swin_large_patch4_window12_384'
        }
        
        model_name_for_timm = timm_model_name_map.get(self.variant, self.variant)

        try:
            self.model = timm.create_model(
                model_name_for_timm,
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=img_size,
                in_chans=in_chans
                # Other Swin-specific params like window_size, embed_dim, depths, num_heads
                # are usually part of the model_name string in timm.
            )
        except TypeError as e:
            # Fallback logic similar to DeiT for img_size and in_chans
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
                raise RuntimeError(f"Failed to create SwinTransformer model {model_name_for_timm} due to unhandled TypeError: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create SwinTransformer model {model_name_for_timm} using timm (general error): {e}")