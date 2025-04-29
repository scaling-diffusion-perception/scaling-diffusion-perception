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
# References:
# - Flying Chairs: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
# --------------------------------------------------------------------------


import os
import torch
import cv2
import numpy as np
from glob import glob
from PIL import Image
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset
import torch.nn.functional as F

# Helper function to read .flo files
def _read_flo(file_name: str) -> np.ndarray:
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Magic number incorrect in {file_name}. Invalid .flo file.")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape((h, w, 2))
    return flow

# Parent class (if it's already defined)
class FlowDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self._flow_list = []
        self._image_list = []
        self.transforms = transforms

    def __len__(self):
        return len(self._flow_list)

    def __getitem__(self, idx):
        raise NotImplementedError

# FlyingChairs class
class FlyingChairs(FlowDataset):
    """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow."""

    def __init__(self, root: str = "/path/to/flow/data", split: str = "train", transforms: Optional[Callable] = None, flow_idx = 0, sota_flag = True) -> None:
        super().__init__(root=root, transforms=transforms)

        self.flow_idx = flow_idx

        root = Path(root)
        images = sorted(glob(str(root / "data" / "*.ppm")))
        flows = sorted(glob(str(root / "data" / "*.flo")))

        split_file_name = "FlyingChairs_train_val.txt"

        if not os.path.exists(root / split_file_name):
            raise FileNotFoundError(
                "The FlyingChairs_train_val.txt file was not found - please download it from the dataset page (see docstring)."
            )

        split_list = np.loadtxt(str(root / split_file_name), dtype=np.int32)

        for i in range(len(flows)):
            split_id = split_list[i]
            if (split == "train" and split_id == 1) or (split == "val" and split_id == 2):
                self._flow_list.append(flows[i])
                self._image_list.append([images[2 * i], images[2 * i + 1]])

    def __getitem__(self, idx):
        # Load the image pair and flow data
        img1_path, img2_path = self._image_list[idx]
        flow_path = self._flow_list[idx]

        img1 = self._load_image(img1_path).permute(1, 2, 0).numpy()  # Convert to HWC for cv2
        img2 = self._load_image(img2_path).permute(1, 2, 0).numpy()
        flow = self._read_flow(flow_path)

        # Use interlinear interpolation instead of padding
        img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Adjust the flow to account for the vertical stretch
        height_scale = 512 / 384
        flow[1, :, :] *= height_scale  # Scale the vertical component of the flow

        # Convert back to tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # Scale the flow appropriately
        flow = flow[self.flow_idx].unsqueeze(0).repeat(3, 1, 1)  # Shape [3, H, W]
        flow = flow / 75  # Adjust based on dataset specifics

        return {
            'rgb': img1,
            'prompt': img2,
            'gt': flow,
            'label': torch.tensor([1002 + self.flow_idx])
        }

    def _load_image(self, file_name: str) -> torch.Tensor:
        """Load image and convert it to a tensor in the range [-1, 1]."""
        img = Image.open(file_name).convert('RGB')
        img = torch.from_numpy(np.array(img)).float() / 255.0  # Convert to [0, 1]
        img = img.permute(2, 0, 1)  # Change from [H, W, C] to [C, H, W]
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img

    def _read_flow(self, file_name: str) -> np.ndarray:
        """Read the .flo optical flow file."""
        return _read_flo(file_name)
