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

import os
import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from pipeline.moe_pipeline import MoEPipeline


def update_model(
        model: MoEPipeline,
        accel = None,
    ):
        
        model.unet.x_embedder.proj.conv_in_1 = _replace_dit_conv_in_multi(model.unet.x_embedder.proj.conv_in_1)
        model.unet.x_embedder.proj.conv_in_2 = _replace_dit_conv_in_multi(model.unet.x_embedder.proj.conv_in_2)
        model.unet.x_embedder.proj.conv_in_3 = _replace_dit_conv_in_multi(model.unet.x_embedder.proj.conv_in_3)
        model.unet.x_embedder.proj.conv_in_4 = _replace_dit_conv_in_multi(model.unet.x_embedder.proj.conv_in_4)
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.unet, model.vae = accel.prepare(model.unet,model.vae)


def _replace_dit_conv_in_multi(conv_layer):
        # replace the first layer to accept 8 in_channels
        _weight = conv_layer.weight.clone()  # [1152, 8, 2, 2]
        _bias = conv_layer.bias.clone()  # [1152]
        _weight = _weight.repeat((1, 3, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.33
        # new conv_in channel
        _n_convin_out_channel = conv_layer.out_channels
        _new_conv_in = Conv2d(
            12, _n_convin_out_channel, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        return _new_conv_in

