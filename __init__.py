from .params import JurdnsPromptParameters
from .data import JurdnsMetadataImageSave

NODE_CLASS_MAPPINGS = {
    "JurdnsPromptParameters": JurdnsPromptParameters,
    "JurdnsMetadataImageSave": JurdnsMetadataImageSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JurdnsPromptParameters": "Jurdns Prompt Parameters",
    "JurdnsMetadataImageSave": "Jurdns Metadata Image Save"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']