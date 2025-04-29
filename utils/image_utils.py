# Copyright 2025 Rahul Ravishankar & Zeeshan Patel, UC Berkeley. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# More information about the method can be found at https://scaling-diffusion-perception.github.io
# --------------------------------------------------------------------------

import torch
from torchvision import transforms
from PIL import Image

def tensor_to_png(tensor, path):
    """
    Converts a PyTorch tensor of shape (1, 3, 512, 512) in the range (0, 1)
    to a PNG file and saves it to the specified path.

    Args:
        tensor (torch.Tensor): A tensor of shape (1, 3, 512, 512) in the range (0, 1).
        path (str): The file path where the PNG should be saved.
    """
    if tensor.min() < 0:
        tensor = tensor * 0.5 + 0.5
    tensor = torch.clip(tensor,0,1)
    tensor = tensor.squeeze(0)
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(path, format='PNG')
