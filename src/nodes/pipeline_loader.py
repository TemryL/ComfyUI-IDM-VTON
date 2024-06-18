import sys
sys.path.append('.')
sys.path.append('..')

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel

from ..idm_vton.unet_hacked_tryon import UNet2DConditionModel
from ..idm_vton.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from ..idm_vton.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from comfy.model_management import get_torch_device
from ...install import WEIGHTS_PATH


DEVICE = get_torch_device()


class PipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_dtype": (("float32", "float16", "bfloat16"), ),
            }
        }
    
    CATEGORY = "ComfyUI-IDM-VTON"
    INPUT_NODE = True
    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_pipeline"
    
    def load_pipeline(self, weight_dtype):
        if weight_dtype == "float32":
            weight_dtype = torch.float32
        elif weight_dtype == "float16":
            weight_dtype = torch.float16
        elif weight_dtype == "bfloat16":
            weight_dtype = torch.bfloat16
        
        noise_scheduler = DDPMScheduler.from_pretrained(
            WEIGHTS_PATH, 
            subfolder="scheduler"
        )
        
        vae = AutoencoderKL.from_pretrained(
            WEIGHTS_PATH,
            subfolder="vae",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        unet = UNet2DConditionModel.from_pretrained(
            WEIGHTS_PATH,
            subfolder="unet",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            WEIGHTS_PATH,
            subfolder="image_encoder",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            WEIGHTS_PATH,
            subfolder="unet_encoder",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        text_encoder_one = CLIPTextModel.from_pretrained(
            WEIGHTS_PATH,
            subfolder="text_encoder",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            WEIGHTS_PATH,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype
        ).requires_grad_(False).eval()
        
        tokenizer_one = AutoTokenizer.from_pretrained(
            WEIGHTS_PATH,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        
        tokenizer_two = AutoTokenizer.from_pretrained(
            WEIGHTS_PATH,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )

        pipe = TryonPipeline.from_pretrained(
            WEIGHTS_PATH,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=weight_dtype,
        )
        pipe.weight_dtype = weight_dtype
        pipe.unet_encoder = unet_encoder
        pipe.enable_sequential_cpu_offload()
        pipe.unet_encoder.to(DEVICE)
        return (pipe, )