import os
import torchvision
import numpy as np
import torch
from PIL import Image


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


def main():
    from src.nodes.pipeline_loader import PipelineLoader
    from src.nodes.idm_vton import IDM_VTON
    
    pipe = PipelineLoader().load_pipeline()[0]
    
    width = 768
    height = 1024
    
    pose_img = Image.open("custom_nodes/ComfyUI-IDM-VTON/input/densepose.jpg")
    garment_img = Image.open("custom_nodes/ComfyUI-IDM-VTON/input/garment.jpg")
    mask_img = Image.open("custom_nodes/ComfyUI-IDM-VTON/input/mask.png")
    model_img = Image.open("custom_nodes/ComfyUI-IDM-VTON/input/model.jpg")
    
    model_prompt = ["model is wearing a shirts"]
    model_negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]
    
    cloth_prompt = ['a photo of shirts']
    cloth_negative_prompt = ['monochrome, lowres, bad anatomy, worst quality, low quality']
    
    strength = 1.0
    num_inference_steps = 30
    guidance_scale = 2.0
    seed = 42
    
    image = IDM_VTON().make_inference(pipe, model_img, garment_img, pose_img, mask_img, height, width, model_prompt, model_negative_prompt, cloth_prompt, cloth_negative_prompt, num_inference_steps, strength, guidance_scale, seed)
    
    x_sample = pil_to_tensor(image)
    torchvision.utils.save_image(x_sample, os.path.join("custom_nodes/ComfyUI-IDM-VTON/result", "result_1.jpg"))
    

if __name__ == "__main__":
    main()
