wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/image_encoder/model.safetensors -P ./models/idm_vton/image_encoder
wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder/model.safetensors -P ./models/idm_vton/text_encoder
wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder_2/model.safetensors -P ./models/idm_vton/text_encoder_2
wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/unet/diffusion_pytorch_model.bin -P ./models/idm_vton/unet
wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/unet_encoder/diffusion_pytorch_model.safetensors -P ./models/idm_vton/unet_encoder
wget -c https://huggingface.co/yisol/IDM-VTON/resolve/main/vae/diffusion_pytorch_model.safetensors -P ./models/idm_vton/vae