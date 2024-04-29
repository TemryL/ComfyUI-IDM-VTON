from .nodes.pipeline_loader import PipelineLoader
from .nodes.idm_vton import IDM_VTON


NODE_CLASS_MAPPINGS = {
    "PipelineLoader": PipelineLoader,
    "IDM-VTON": IDM_VTON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipelineLoader": "Load IDM-VTON Pipeline",
    "IDM-VTON": "Run IDM-VTON Inference",
}