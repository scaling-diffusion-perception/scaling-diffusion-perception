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

import numpy as np
import torch
import math

def hsv_to_rgb(h, s, v):
    """
    Converts HSV to RGB format. The input tensors h, s, and v should be in the range [0, 1].
    """
    h = h * 6.0  # Scale hue to [0, 6]
    c = v * s
    x = c * (1 - torch.abs((h % 2) - 1))
    m = v - c

    # c, x, and m should all have the shape (512, 512)
    z = torch.zeros_like(h)
    
    # Compute each RGB channel based on hue range
    r = torch.where((h < 1), c, torch.where((h < 2), x, torch.where((h < 4), z, torch.where((h < 5), x, c))))
    g = torch.where((h < 1), x, torch.where((h < 3), c, torch.where((h < 4), x, z)))
    b = torch.where((h < 2), z, torch.where((h < 3), x, torch.where((h < 5), c, x)))
    
    # Combine channels and add m for brightness adjustment
    rgb = torch.stack([r + m, g + m, b + m], dim=0)  # (3, 512, 512)

    return rgb  # Return in the format (3, 512, 512)

def create_colored_map(flow_x, flow_y, gt_x, gt_y):
    """
    Creates a color-coded optical flow map using predicted and ground truth flow components,
    where zero magnitude is mapped to white.
    
    Args:
        flow_x, flow_y: Predicted flow components in the x and y directions.
        gt_x, gt_y: Ground truth flow components in the x and y directions.
    
    Returns:
        Colored prediction and ground truth flow maps in shape (1, 3, 512, 512) in the range (0, 1).
    """
    # Remove batch and channel dimensions
    flow_x, flow_y = flow_x.squeeze(0)[0], flow_y.squeeze(0)[0]
    gt_x, gt_y = gt_x.squeeze(0)[0], gt_y.squeeze(0)[0]

    # Scale predictions
    _, alpha_x, beta_x, alpha_y, beta_y = calculate_epe_with_fit(flow_x, gt_x, flow_y, gt_y)
    flow_x = alpha_x * flow_x + beta_x
    flow_y = alpha_y * flow_y + beta_y

    # Calculate angle and magnitude for predicted flow
    angle_pred = torch.atan2(flow_y, flow_x)
    magnitude_pred = torch.sqrt(flow_x ** 2 + flow_y ** 2)

    # Normalize magnitude to (0, 1)
    magnitude_pred = torch.clip(magnitude_pred / magnitude_pred.max(), 0, 1)
    
    # HSV color map for predicted flow, inspired by color wheel
    hue_pred = (angle_pred + math.pi) / (2 * math.pi)  # Map angle [-π, π] to hue [0, 1]
    saturation_pred = magnitude_pred  # Low magnitudes have low saturation, approaching white
    value_pred = torch.ones_like(hue_pred) * 0.9  # Keep brightness high overall

    # Convert HSV to RGB for predicted flow
    colored_map_pred = hsv_to_rgb(hue_pred, saturation_pred, value_pred).unsqueeze(0)  # (1, 3, 512, 512)

    # Repeat for ground truth flow
    angle_gt = torch.atan2(gt_y, gt_x)
    magnitude_gt = torch.sqrt(gt_x ** 2 + gt_y ** 2)
    magnitude_gt = torch.clip(magnitude_gt / magnitude_gt.max(), 0, 1)
    hue_gt = (angle_gt + math.pi) / (2 * math.pi)
    saturation_gt = magnitude_gt
    value_gt = torch.ones_like(hue_gt) * 0.9
    colored_map_gt = hsv_to_rgb(hue_gt, saturation_gt, value_gt).unsqueeze(0)  # (1, 3, 512, 512)

    return colored_map_pred, colored_map_gt

def calculate_epe_with_fit(predicted_flow_x, ground_truth_flow_x, predicted_flow_y, ground_truth_flow_y):
    """
    Calculate the optimal alpha and beta using least squares fitting for both x and y directions,
    then compute the End-Point Error (EPE) for optical flow.
    
    Parameters:
    - predicted_flow_x (torch.Tensor): Predicted flow in the x direction, shape (512, 512), range (0,1).
    - ground_truth_flow_x (torch.Tensor): Ground truth flow in the x direction, shape (512, 512).
    - predicted_flow_y (torch.Tensor): Predicted flow in the y direction, shape (512, 512), range (0,1).
    - ground_truth_flow_y (torch.Tensor): Ground truth flow in the y direction, shape (512, 512).
    
    Returns:
    - epe (float): The EPE error after fitting.
    - alpha_x, beta_x (float): The scaling and shifting factors for the x direction.
    - alpha_y, beta_y (float): The scaling and shifting factors for the y direction.
    """
    
    # Flatten the flow maps for linear fitting
    pred_x_flat = predicted_flow_x.flatten()
    gt_x_flat = ground_truth_flow_x.flatten()
    pred_y_flat = predicted_flow_y.flatten()
    gt_y_flat = ground_truth_flow_y.flatten()
    
    # Perform least squares fitting to find optimal alpha and beta for x
    A_x = np.vstack([pred_x_flat, np.ones_like(pred_x_flat)]).T
    alpha_x, beta_x = np.linalg.lstsq(A_x, gt_x_flat, rcond=None)[0]
    
    # Perform least squares fitting to find optimal alpha and beta for y
    A_y = np.vstack([pred_y_flat, np.ones_like(pred_y_flat)]).T
    alpha_y, beta_y = np.linalg.lstsq(A_y, gt_y_flat, rcond=None)[0]
    
    # Scale the predicted flow maps using the obtained alpha and beta
    scaled_predicted_flow_x = alpha_x * predicted_flow_x + beta_x
    scaled_predicted_flow_y = alpha_y * predicted_flow_y + beta_y
    
    # Calculate the error
    error_x = scaled_predicted_flow_x - ground_truth_flow_x
    error_y = scaled_predicted_flow_y - ground_truth_flow_y
    
    # Compute the overall End-Point Error (EPE)
    # print((error_x** 2).max(), (error_y** 2).max(), (error_x** 2).min(), (error_y** 2).min())
    epe = torch.sum(torch.sqrt(error_x ** 2 + error_y ** 2)) / error_x.numel()
    
    return epe, alpha_x, beta_x, alpha_y, beta_y

def calculate_epe_no_fit(predicted_flow_x, ground_truth_flow_x, predicted_flow_y, ground_truth_flow_y):
    """
    Calculate the End-Point Error (EPE) for optical flow.
    
    Parameters:
    - predicted_flow_x (torch.Tensor): Predicted flow in the x direction, shape (512, 512).
    - ground_truth_flow_x (torch.Tensor): Ground truth flow in the x direction, shape (512, 512).
    - predicted_flow_y (torch.Tensor): Predicted flow in the y direction, shape (512, 512).
    - ground_truth_flow_y (torch.Tensor): Ground truth flow in the y direction, shape (512, 512).
    
    Returns:
    - epe (float): The EPE error after fitting.
    """
    
    scaled_predicted_flow_x = predicted_flow_x
    scaled_predicted_flow_y = predicted_flow_y
    
    # Calculate the error
    error_x = scaled_predicted_flow_x - ground_truth_flow_x
    error_y = scaled_predicted_flow_y - ground_truth_flow_y
    
    epe = torch.sum(torch.sqrt(error_x ** 2 + error_y ** 2)) / error_x.numel()
    return epe
