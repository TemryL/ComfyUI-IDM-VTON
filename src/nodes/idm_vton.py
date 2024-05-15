import sys
sys.path.append('.')
sys.path.append('..')

import torch
from torchvision import transforms
from comfy.model_management import get_torch_device


DEVICE = get_torch_device()
MAX_RESOLUTION = 16384


class IDM_VTON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "human_img": ("IMAGE",),
                "pose_img": ("IMAGE",),
                "mask_img": ("IMAGE",),
                "garment_img": ("IMAGE",),
                "garment_description": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
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
    
    def preprocess_images(self, human_img, garment_img, pose_img, mask_img, height, width):
        human_img = human_img.squeeze().permute(2,0,1)
        garment_img = garment_img.squeeze().permute(2,0,1)
        pose_img = pose_img.squeeze().permute(2,0,1)
        mask_img = mask_img.squeeze().permute(2,0,1)
        
        human_img = transforms.functional.to_pil_image(human_img)  
        garment_img = transforms.functional.to_pil_image(garment_img)  
        pose_img = transforms.functional.to_pil_image(pose_img)  
        mask_img = transforms.functional.to_pil_image(mask_img)
        
        human_img = human_img.convert("RGB").resize((width, height))
        garment_img = garment_img.convert("RGB").resize((width, height))
        mask_img = mask_img.convert("RGB").resize((width, height))
        pose_img = pose_img.convert("RGB").resize((width, height))
        
        return human_img, garment_img, pose_img, mask_img
    
    def make_inference(self, pipeline, human_img, garment_img, pose_img, mask_img, height, width, garment_description, negative_prompt, num_inference_steps, strength, guidance_scale, seed):
        human_img, garment_img, pose_img, mask_img = self.preprocess_images(human_img, garment_img, pose_img, mask_img, height, width)
        tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        with torch.no_grad():
            # Extract the images
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    prompt = "model is wearing " + garment_description
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipeline.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = ["a photo of " + garment_description]
                    negative_prompt = [negative_prompt]
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipeline.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(DEVICE, pipeline.dtype)
                    garment_tensor = tensor_transfrom(garment_img).unsqueeze(0).to(DEVICE, pipeline.dtype)
                    
                    images = pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        generator=torch.Generator(DEVICE).manual_seed(seed),
                        strength=strength,
                        pose_img=pose_img,
                        text_embeds_cloth=prompt_embeds_c,
                        cloth=garment_tensor,
                        mask_image=mask_img,
                        image=human_img, 
                        height=height,
                        width=width,
                        ip_adapter_image=garment_img,
                        guidance_scale=guidance_scale,
                    )[0]
                    
                    images = [transforms.ToTensor()(image) for image in images]
                    images = [image.permute(1,2,0) for image in images]
                    images = torch.stack(images)
                    return (images, )