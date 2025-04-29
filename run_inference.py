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

import argparse
import os
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pipeline.moe_pipeline import MoEPipeline

from src.util.config_util import (
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)

from src.dataset import DatasetMode
from src.dataset import FlyingChairs, GestaltData, HypersimDataset
from accelerate import Accelerator
import sys

from utils.image_utils import tensor_to_png
from utils.flow_utils import create_colored_map
from utils.model_utils import update_model
from utils.depth_utils import calculate_depth_with_fit, colorize_depth_maps

from models import find_model
from pipeline.util.ensemble import ensemble_depth


if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Inference your model!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dit_moe_generalist.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save results"
    )
    parser.add_argument(
        "--path_to_sd",
        type=str,
        default=None,
        help="/path/to/stable-diffusion-2/",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="name of job (a1_dsratio-0.5_res-256)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="/path/to/DiT_a4_0080000.pt",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="DiT_a_1",
    )
    parser.add_argument(
        "--num_exps",
        type=int,
        default=0,
        help="16",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="16",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="16",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="depth",
        help="depth, flow, segment",
    )
    parser.add_argument(
        "--generate_vis",
        type=int,
        default=5,
        help="number of visualizations to generate",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/dataset",
        help="path to respective data folder",
    )
    parser.add_argument(
        "--data_ls",
        type=str,
        default="/path/to/data/ls.txt",
        help="path to data list",
    )
    parser.add_argument(
        "--color",
        action='store_true',
        help="colors for depth map included",
    )

    accelerator = Accelerator()
    args = parser.parse_args()
    output_dir = args.output_dir

    # -------------------- Initialization --------------------
    cfg = recursive_load_config(args.config)
    # Full job name
    pure_job_name = os.path.basename(args.config).split(".")[0]
    job_name = args.job_name

    # Output dir
    out_dir_run = os.path.join(output_dir, job_name)
    os.makedirs(out_dir_run, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available()
    device = accelerator.device

    # -------------------- Data --------------------
    if args.task == "flow":
        flow_dx_dataset = FlyingChairs(root=args.data_path, flow_idx=0, sota_flag=True, split='val')
        flow_dy_dataset = FlyingChairs(root=args.data_path, flow_idx=1, sota_flag=True, split='val')
        data_loader_dx = DataLoader(
            flow_dx_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
        )
        data_loader_dy = DataLoader(
            flow_dy_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
        )
        data_loader_dx = accelerator.prepare(data_loader_dx)
        data_loader_dy = accelerator.prepare(data_loader_dy)

    elif args.task == "depth":
        val_loaders: List[DataLoader] = []
        depth_transform: DepthNormalizerBase = get_depth_normalizer(
            cfg_normalizer=cfg.depth_normalization
        )
        hypersim_dataset = HypersimDataset(
            mode=DatasetMode.TRAIN,
            filename_ls_path=args.data_ls,
            dataset_dir=args.data_path,
            disp_name="Hypersim",
            depth_transform=depth_transform,
            augmentation_args=cfg.augmentation,
            base_data_dir=args.data_path,
            class_embed=1000,
            sota_flag=True,
        )
        data_loader = DataLoader(
            hypersim_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
        )
        data_loader = accelerator.prepare(data_loader)

    elif args.task == "segment":
        pix2gestalt_dataset = GestaltData(args.data_path, validation=True, sota_flag=True)
        data_loader = DataLoader(
            pix2gestalt_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
        )
        data_loader = accelerator.prepare(data_loader)
    
    else:
        print("TASK IS NOT KNOWN")
        sys.exit()

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    _pipeline_kwargs['model_path'] = args.model_path
    _pipeline_kwargs['model_type'] = args.model_type
    _pipeline_kwargs['num_exps'] = args.num_exps
    _pipeline_kwargs['top_k'] = args.top_k
    _pipeline_kwargs['beta_schedule'] = getattr(cfg.pipeline.kwargs, 'beta_schedule', 'linear')
    _pipeline_kwargs['extra_routing_util'] = None

    model = MoEPipeline.from_pretrained(
            args.path_to_sd, True, **_pipeline_kwargs
        )
    update_model(model, accelerator)
    state_dict = find_model(args.model_path)
    missing_keys, unexpected_keys = model.unet.load_state_dict(state_dict)
    print(f"Missing Keys: {missing_keys}")
    print(f"Unexpected Keys: {unexpected_keys}")
    model.unet.eval()

    # -------------------- Visualization Loop --------------------
    from src.util.seeding import generate_seed_sequence
    import numpy as np
    import torch.nn.functional as F
    from einops import rearrange
    from torchvision.transforms import InterpolationMode, Resize
    from tqdm import tqdm
    import time

    val_init_seed = cfg.validation.init_seed
    val_seed_ls = generate_seed_sequence(val_init_seed, args.generate_vis * args.num_samples)
    idx = 0
    resize_transform = Resize(
                size=(512, 512), interpolation=InterpolationMode.NEAREST_EXACT
            )
    
    if args.task == "depth" or args.task == "segment":
        for batch in tqdm(data_loader, total=args.generate_vis, desc="Visualization"):
            idx += 1


            rgb = batch['rgb'].to(device).to(torch.float32)
            rgb = resize_transform(rgb)
            prompt = batch['prompt'].to(device).to(torch.float32)
            prompt = resize_transform(prompt)
            gt = batch['gt'].to(device).to(torch.float32)
            gt = resize_transform(gt)
            label = batch['label'].to(device).int().squeeze(0)


            # Predict depth
            if args.task == "depth":
                joint_preds = torch.zeros((args.num_samples, 1, 512, 512)).to(device=device)
                for i_ in range(args.num_samples):
                    # Random number generator
                    seed = val_seed_ls.pop()
                    if seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(seed)
                    pred = model.single_infer(
                        rgb,
                        prompt,
                        label,
                        num_inference_steps=cfg.validation.denoising_steps,
                        generator=generator,
                        show_pbar=False,
                        device=accelerator.device
                    )
                    pred = pred.mean(dim=1)
                    joint_preds[i_] = pred
                depth_pred,_ = ensemble_depth(
                    joint_preds,
                    scale_invariant=True,
                    shift_invariant=True,
                    max_res=50,
                    **({}),
                )
                depth_pred = calculate_depth_with_fit(depth_pred[:,0].squeeze().cpu(), gt[:,0].squeeze().cpu())
                depth_pred = torch.clamp(depth_pred, min=-1, max=1)
                if args.color:
                    pred = colorize_depth_maps((depth_pred+1)/2, 0,  1)
                    gt = colorize_depth_maps((gt.mean(dim=1).squeeze().cpu()+1)/2, 0, 1)
                else:
                    pred = depth_pred.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,3,1,1)

                tensor_to_png(gt.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_gt.png")
                tensor_to_png(pred.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_pred.png")
                tensor_to_png(rgb.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_rgb.png")
                tensor_to_png(prompt.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_prompt.png")

                # Generate Toy Data
                tensor_to_png(rgb.cpu(), f"toy_data/{args.job_name}_{args.task}_{idx}_rgb.png")
                tensor_to_png(prompt.cpu(), f"toy_data/{args.job_name}_{args.task}_{idx}_prompt.png")
            
            # Predict Segment
            if args.task == "segment":
                joint_preds = torch.zeros((args.num_samples, 3, 512, 512)).to(device=device)
                for i_ in range(args.num_samples):
                    # Random number generator
                    seed = val_seed_ls.pop()
                    if seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(seed)
                    pred = model.single_infer(
                        rgb,
                        prompt,
                        label,
                        num_inference_steps=cfg.validation.denoising_steps,
                        generator=generator,
                        show_pbar=False,
                        device=accelerator.device
                    )
                    joint_preds[i_] = pred.squeeze()
                pred = joint_preds.median(dim=0, keepdim=True).values
                pred = torch.clamp(pred, min=-1, max=1)
                tensor_to_png(gt.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_gt.png")
                tensor_to_png(pred.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_pred.png")
                tensor_to_png(rgb.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_rgb.png")
                tensor_to_png(prompt.cpu(), f"samples/{args.job_name}_{args.task}_{idx}_prompt.png")

                # Generate Toy Data
                tensor_to_png(rgb.cpu(), f"toy_data/{args.job_name}_{args.task}_{idx}_rgb.png")
                tensor_to_png(prompt.cpu(), f"toy_data/{args.job_name}_{args.task}_{idx}_prompt.png")

            if idx == args.generate_vis:
                break
    
    elif args.task == "flow":
        for batch_dx,batch_dy in tqdm(zip(data_loader_dx, data_loader_dy), total=args.generate_vis, desc="Visualization"):

            idx += 1
            val_seed_ls_1 = val_seed_ls.copy()
            val_seed_ls_2 = val_seed_ls.copy()

            rgb = batch_dx['rgb'].to(device).to(torch.float32)
            rgb = resize_transform(rgb)
            prompt = batch_dx['prompt'].to(device).to(torch.float32)
            prompt = resize_transform(prompt)
            gt_dx = batch_dx['gt'].to(device).to(torch.float32)
            gt_dx = resize_transform(gt_dx)
            label_dx = batch_dx['label'].to(device).int().squeeze(0)
            gt_dy = batch_dy['gt'].to(device).to(torch.float32)
            gt_dy = resize_transform(gt_dy)
            label_dy = batch_dy['label'].to(device).int().squeeze(0)
            # Predict Flow
            joint_preds = torch.zeros((args.num_samples, 1, 512, 512)).to(device=device)
            for i_ in range(args.num_samples):
                # Random number generator
                seed = val_seed_ls_1.pop()
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(seed)
                pred = model.single_infer(
                    rgb,
                    prompt,
                    label_dx,
                    num_inference_steps=cfg.validation.denoising_steps,
                    generator=generator,
                    show_pbar=False,
                    device=accelerator.device
                )
                pred = pred.mean(dim=1)
                joint_preds[i_] = pred
            pred_dx = joint_preds.mean(dim=0,keepdim=True).repeat(1,3,1,1)


            for i_ in range(args.num_samples):
                # Random number generator
                seed = val_seed_ls_2.pop()
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(seed)
                pred = model.single_infer(
                    rgb,
                    prompt,
                    label_dy,
                    num_inference_steps=cfg.validation.denoising_steps,
                    generator=generator,
                    show_pbar=False,
                    device=accelerator.device
                )
                pred = pred.mean(dim=1)
                joint_preds[i_] = pred
            pred_dy = joint_preds.mean(dim=0,keepdim=True).repeat(1,3,1,1)

            colored_map_pred, colored_map_gt = create_colored_map(pred_dx.cpu(), pred_dy.cpu(), gt_dx.cpu(), gt_dy.cpu())
            tensor_to_png(colored_map_pred, f"samples/{args.job_name}_{args.task}_{idx}_pred.png")
            tensor_to_png((rgb+prompt)/2, f"samples/{args.job_name}_{args.task}_{idx}_rgb.png")
            tensor_to_png(colored_map_gt, f"samples/{args.job_name}_{args.task}_{idx}_gt.png")
            
            # Generate Toy Data
            tensor_to_png(rgb, f"toy_data/{args.job_name}_{args.task}_{idx}_rgb.png")
            tensor_to_png(prompt, f"toy_data/{args.job_name}_{args.task}_{idx}_prompt.png")
            if idx == args.generate_vis:
                break
