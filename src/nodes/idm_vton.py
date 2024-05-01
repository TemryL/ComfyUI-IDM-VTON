import sys
sys.path.append('.')
sys.path.append('..')

import torch
from torchvision import transforms
from transformers import CLIPImageProcessor
from comfy.model_management import get_torch_device


DEVICE = get_torch_device()
MAX_RESOLUTION=16384


class IDM_VTON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "model_img": ("IMAGE",),
                "pose_img": ("IMAGE",),
                "mask_img": ("IMAGE",),
                "garment_img": ("IMAGE",),
                "model_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model_negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "garment_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "garment_negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 768, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION}),
                "num_inference_steps": ("INT", {"default": 30}),
                "guidance_scale": ("FLOAT", {"default": 2.0}),
                "strength": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "make_inference"
    CATEGORY = "ComfyUI-IDM-VTON"
    
    def preprocess_images(self, model_img, garment_img, pose_img, mask_img, width, height):        
        model_img = model_img.permute(0, 3, 1, 2)
        garment_img = garment_img.permute(0, 3, 1, 2)
        pose_img = pose_img.permute(0, 3, 1, 2)
        mask_img = mask_img.permute(0, 3, 1, 2)
        
        transform = transforms.Compose(
            [
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        clip_processor = CLIPImageProcessor()
        image_embeds = clip_processor(images=garment_img, return_tensors="pt").pixel_values[0].unsqueeze(0)
        
        # model_img = model_img.resize((width, height))
        model_img = transform(model_img)
        model_img = (model_img + 1.0) / 2.0
        model_img = model_img.to(DEVICE)
        
        garment_img = transform(garment_img)
        garment_img = garment_img.to(DEVICE)
        
        pose_img = transform(pose_img)
        pose_img = pose_img.to(DEVICE)
        
        # mask_img = mask_img.resize((width, height))
        mask_img = mask_img[:,:1,:,:]
        mask_img = mask_img.to(DEVICE)
        
        return model_img, garment_img, pose_img, mask_img, image_embeds
    
    def make_inference(self, pipeline, model_img, garment_img, pose_img, mask_img, height, width, model_prompt, model_negative_prompt, garment_prompt, garment_negative_prompt, num_inference_steps, strength, guidance_scale, seed):
        model_img, garment_img, pose_img, mask_img, image_embeds = self.preprocess_images(model_img, garment_img, pose_img, mask_img, width, height)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipeline.encode_prompt(
                        model_prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=model_negative_prompt,
                    )
                    
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipeline.encode_prompt(
                        garment_prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=garment_negative_prompt,
                    )
                    
                    print("{:.2f}".format(torch.cuda.memory_allocated(DEVICE)*1e-9))
                    
                    images = pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        generator=torch.Generator(pipeline.device).manual_seed(seed),
                        strength=strength,
                        pose_img=pose_img,
                        text_embeds_cloth=prompt_embeds_c,
                        cloth=garment_img,
                        mask_image=mask_img,
                        image=model_img, 
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        ip_adapter_image=image_embeds,
                    )[0]
                    
                    print(type(images))
                    print(type(images[0]))
                    
                    return (images[0], )