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
# - pix2gestalt: https://github.com/cvlab-columbia/pix2gestalt
# --------------------------------------------------------------------------

import os
import importlib
from pathlib import Path
from abc import abstractmethod
import cv2
import math
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class PRNGMixin(object):
    """
    Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing.
    """
    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


"""A modified image folder class
We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class GestaltData(Dataset):
    def __init__(self,
            root_dir,
            validation=False, 
            is_trainsize_ablation = False,
            sota_flag = False
            ) -> None:
        self.root_dir = Path(root_dir)

        self.occlusion_root = os.path.join(root_dir, 'occlusion')
        self.whole_root = os.path.join(root_dir, 'whole')
        self.whole_mask_root = os.path.join(root_dir, 'whole_mask')
        self.visible_mask_root = os.path.join(root_dir, 'visible_object_mask')
        self.sota_flag = sota_flag

        all_occlusion_paths = sorted(make_dataset(self.occlusion_root))
        all_whole_paths = sorted(make_dataset(self.whole_root))
        all_whole_mask_paths = sorted(make_dataset(self.whole_mask_root))
        all_visible_mask_paths = sorted(make_dataset(self.visible_mask_root))

        total_objects = len(all_occlusion_paths)
        print("Total number of samples: %d" % total_objects)
        # val_start = math.floor(total_objects / 100. * 99.)
        val_start = math.floor(0)
        if validation:
            # Last 1% as validation
            self.occlusion_paths = all_occlusion_paths[val_start:]
            self.whole_paths = all_whole_paths[val_start:]
            self.whole_mask_paths = all_whole_mask_paths[val_start:]
            self.visible_mask_paths = all_visible_mask_paths[val_start:]
        else:
            # First 99% as training
            self.occlusion_paths = all_occlusion_paths[:val_start]
            self.whole_paths = all_whole_paths[:val_start]
            self.whole_mask_paths = all_whole_mask_paths[:val_start]
            self.visible_mask_paths = all_visible_mask_paths[:val_start]
        print('============= length of %s dataset %d =============' % ("validation" if validation else "training", self.__len__()))

    def upscale_image(self, image_tensor, size=(512, 512)):
        return F.interpolate(image_tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
    
    def upscale_image_nearest(self, image_tensor, size=(512, 512)):
        return F.interpolate(image_tensor.unsqueeze(0), size=size, mode='nearest').squeeze(0)

    def __len__(self):
        return len(self.occlusion_paths)

    def __getitem__(self, index):
        data = {}
    
        visible_mask_path = self.visible_mask_paths[index]
        occlusion_image_path =  self.occlusion_paths[index]
        whole_image_path = self.whole_paths[index]

        occluded_object_image = read_rgb(occlusion_image_path)
        whole_object_image = read_rgb(whole_image_path)
        visible_mask = read_mask(visible_mask_path)

        rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
        rgb_visible_mask[:,:,0] = visible_mask
        rgb_visible_mask[:,:,1] = visible_mask
        rgb_visible_mask[:,:,2] = visible_mask

        occluded_object_image = self.process_image(occluded_object_image) # input occlusion image
        rgb_visible_mask = self.process_image(rgb_visible_mask) # input visible (modal) mask
        whole_object_image = self.process_image(whole_object_image) # target whole (amodal) image

        if self.sota_flag:
            data["rgb"] = torch.from_numpy(occluded_object_image).permute(2,0,1)
            data["rgb"] = self.upscale_image(data["rgb"])
            data["prompt"] = torch.from_numpy(rgb_visible_mask).permute(2,0,1)
            data["prompt"] = self.upscale_image_nearest(data["prompt"])
            data["gt"] = torch.from_numpy(whole_object_image).permute(2,0,1)
            data["gt"] = self.upscale_image(data["gt"])
            data["label"] = torch.tensor([1001])
        else:
            data["image_cond"] = torch.from_numpy(occluded_object_image).permute(2,0,1)
            data["visible_mask_cond"] = torch.from_numpy(rgb_visible_mask).permute(2,0,1)
            data["image_target"] = torch.from_numpy(whole_object_image).permute(2,0,1)
            data["class_label"] = torch.tensor([1001])
        return data

    def process_image(self, input_im):
        input_im = input_im.astype(float) / 255. # [0, 255] to [0., 1.]
        normalized_image = input_im * 2 - 1 # [0, 1] to [-1, 1]
        return normalized_image

def read_mask(file_path):
    """
    In:
        file_path: Path to binary mask png image.
    Out:
        binary mask as np array [height, width].
    Purpose:
        Read in a mask image.
    """
    return cv2.imread(file_path, -1)

def read_rgb(file_path):
    """
    In:
        file_path: Color image png to read.
    Out:
        RGB image as np array [height, width, 3], each value in range [0, 255]. Color channel in the order RGB.
    Purpose:
        Read in a color image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
