# Author: Bingxin Ke
# Last modified: 2024-02-22

import torch


def get_loss(loss_name, **kwargs):
    # if "silog_mse" == loss_name:
    #     criterion = SILogMSELoss(**kwargs)
    # elif "silog_rmse" == loss_name:
    #     criterion = SILogRMSELoss(**kwargs)
    # elif "mse_loss" == loss_name:
    #     criterion = torch.nn.MSELoss(**kwargs)
    # elif "l1_loss" == loss_name:
    #     criterion = torch.nn.L1Loss(**kwargs)
    # elif "l1_loss_with_mask" == loss_name:
    #     criterion = L1LossWithMask(**kwargs)
    # elif "mean_abs_rel" == loss_name:
    #     criterion = MeanAbsRelLoss()
    # elif "edm" == loss_name:
    #     criterion = EDMLoss(**kwargs)
    # else:
    #     raise NotImplementedError
    criterion = torch.nn.MSELoss(**kwargs)
    return criterion


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, reduction=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.reduction = reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        # check sigma shape
        rnd_normal = torch.randn([depth_pred.shape[0], 1, 1, 1], device=depth_pred.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # compute weight for loss scaling
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)

        # compute loss
        loss = weight * ((depth_pred - depth_gt) ** 2)

        # apply valid_mask
        if valid_mask is not None:
            loss = loss * valid_mask
            n_valid = valid_mask.sum(dim=(-1, -2))
        else:
            n_valid = depth_gt.shape[-1] * depth_gt.shape[-2]

        # compute per-sample loss
        loss_per_sample = loss.sum(dim=(-1, -2)) / n_valid

        # reduce loss over the batch if required
        if self.reduction:
            loss = loss_per_sample.mean()
        else:
            loss = loss_per_sample

        return loss
