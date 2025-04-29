# Author: Bingxin Ke
# Last modified: 2024-03-12

import logging
import os
import sys
import wandb
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

def config_logging(cfg_logging, out_dir=None, accelerator=None):

    if accelerator and not accelerator.is_main_process:
        return  # Only rank 0 should configure logging
    
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(cfg_logging["format"])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


class MyTrainingLogger:
    """Tensorboard + wandb logger"""

    writer: SummaryWriter
    is_initialized = False

    def __init__(self, accelerator=None) -> None:
        self.accelerator = accelerator
        pass

    def set_dir(self, tb_log_dir):
        if self.accelerator and not self.accelerator.is_main_process:
            return  # Only rank
        if self.is_initialized:
            raise ValueError("Do not initialize writer twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dic(self, scalar_dic, global_step, walltime=None):
        if self.accelerator and not self.accelerator.is_main_process:
            return  # Only rank
        for k, v in scalar_dic.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
        return


# global instance
# tb_logger = MyTrainingLogger()


# -------------- wandb tools --------------
def init_wandb(enable: bool, accelerator=None, **kwargs):
    # Check if we're in a distributed setup and if the current process is the rank-zero process
    if accelerator is not None and not accelerator.is_main_process:
        return None

    if enable:
        run = wandb.init(sync_tensorboard=True, **kwargs)
    else:
        run = wandb.init(mode="disabled")
    return run

def log_slurm_job_id(step, accelerator=None, tb_logging=None):
    # Ensure only rank-zero logs the SLURM job ID
    if accelerator is not None and not accelerator.is_main_process:
        return
    # if not tb_logging:
    #     global tb_logger
    else:
        tb_logger = tb_logging
    _jobid = os.getenv("SLURM_JOB_ID")
    if _jobid is None:
        _jobid = -1
    tb_logger.writer.add_scalar("job_id", int(_jobid), global_step=step)
    logging.debug(f"Slurm job_id: {_jobid}")


def load_wandb_job_id(out_dir, accelerator=None):
    # Ensure only rank-zero process loads the wandb job ID
    if accelerator is not None and not accelerator.is_main_process:
        return None

    with open(os.path.join(out_dir, "WANDB_ID"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def save_wandb_job_id(run, out_dir, accelerator=None):
    # Ensure only rank-zero process saves the wandb job ID
    if accelerator is not None and not accelerator.is_main_process:
        return

    with open(os.path.join(out_dir, "WANDB_ID"), "w+") as f:
        f.write(run.id)


def eval_dic_to_text(val_metrics: dict, dataset_name: str, sample_list_path: str):
    eval_text = f"Evaluation metrics:\n\
     on dataset: {dataset_name}\n\
     over samples in: {sample_list_path}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text
