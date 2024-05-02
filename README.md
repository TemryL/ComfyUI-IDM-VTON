# ComfyUI-IDM-VTON
ComfyUI adaptation of [IDM-VTON](https://github.com/yisol/IDM-VTON).

![workflow](./assets/workflow.png)

# Installation

:warning: Current implementation requires GPU with at least 16GB of VRAM :warning:

### Using [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager):

- Look for ```ComfyUI-IDM-VTON```, and be sure the author is ```TemryL```. Install it.

### Manually:
- Clone this repo into `custom_nodes` folder in ComfyUI and install the dependencies.
```bash
cd custom_nodes
git clone https://github.com/TemryL/ComfyUI-IDM-VTON.git
pip install -r requirements.txt 
```

# Mask Generation
The workflow provided above uses [ComfyUI Segment Anything](https://github.com/storyicon/comfyui_segment_anything) to generate the image mask.

# DensePose Estimation
DensePose estimation is performed using [ComfyUI's ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux).

# Contribute
Thanks for your interest in contributing to the source code! We welcome help from anyone and appreciate every contribution, no matter how small!

If you're ready to contribute, please create a fork, make your changes, commit them, and then submit a pull request for review. We'll consider it for integration into the main code base.

# Credits
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [IDM-VTON](https://github.com/yisol/IDM-VTON)
- [ComfyUI Segment Anything](https://github.com/storyicon/comfyui_segment_anything)
- [ComfyUI's ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux)

# License
Under [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
