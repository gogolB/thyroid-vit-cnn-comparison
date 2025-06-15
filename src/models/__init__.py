# This file ensures that the submodules (and their __init__.py files) are loaded,
# which should trigger model registration.

# Import cnn models so they get registered
import src.models.cnn

# Import vit models so they get registered
import src.models.vit

# Import ensemble models if they have their own registration
# import src.models.ensemble

# Import hybrid models if they have their own registration
# import src.models.hybrid

# Optionally, define __all__ if you want to control what 'from src.models import *' imports
# __all__ = [] # Add model class names or submodule names here