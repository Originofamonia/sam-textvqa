import argparse
import logging
import os
import sys
import random
from io import open
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

pwd = os.getcwd()
sys.path.insert(0, pwd)

from evaluator import Evaluator
from sam.sa_m4c import SAM4C, BertConfig
from sam.task_utils import (clip_gradients, forward_model,
                            get_optim_scheduler, load_datasets)
from tools.registry import registry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_config():
    # load command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=444, help="Random seed for reproducibility"
    )
    parser.add_argument("--config", type=str, default='configs/train-tvqa-eval-tvqa-c3.yml',
        help="Experiment configuration file")

    parser.add_argument(
        "--tag", type=str, help="Experiment folder name", default="tvqa"
    )

    parser.add_argument("--pretrained_eval", default="", help="Path of pre-trained checkpoint")
    opt = parser.parse_args()

    # Load configuration
    with open(opt.config, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    # Todo: Move below code to another function
    # Reproducibility seeds
    seed = task_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("-" * 20 + "Command Line Config: " + "-" * 20)
    print(pprint(vars(opt)))
    logger.info("-" * 20 + "Task File Config: " + "-" * 20)
    print(pprint(task_cfg))

    # Build save path
    save_path = os.path.join(task_cfg["output_dir"], opt.tag)
    if not os.path.exists(save_path) and opt.pretrained_eval == "":
        os.makedirs(save_path)

    # Dump all configs
    with open(os.path.join(save_path, "command.txt"), "w") as f:
        print(f"Command Line: \n {str(vars(opt))} \n \n", file=f)
        print(f"Config File: \n {str(vars(task_cfg))} \n \n", file=f)

    # Add all configs to registry
    registry.update(vars(opt))
    registry.update(task_cfg)

    return task_cfg, opt, save_path


def main():
    task_cfg, opt, save_path = get_config()
    checkpoint_path = os.path.join(save_path, "best_model.pt")
    base_lr = task_cfg["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Numer of GPUs: {n_gpu}")

    dataloaders = load_datasets(task_cfg, ["train", "val", "test"])

    mmt_config = BertConfig.from_dict(task_cfg["SA-M4C"])
    text_bert_config = BertConfig.from_dict(task_cfg["TextBERT"])
    model = SAM4C(mmt_config, text_bert_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler = get_optim_scheduler(
        task_cfg, optimizer_grouped_parameters, base_lr
    )
    start_iter_id, global_step, start_epoch = 0, 0, 0
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # When running only evaluation
    if opt.pretrained_eval != "":
        logger.info(
            f"Dumping Evaluation results at: {os.path.dirname(opt.pretrained_eval)}"
        )
        return opt.pretrained_eval, model, dataloaders

    # This validation score is used for model-saving.
    best_val_step, best_val_score = -1, -1
    loss_values, score_values = [], []
    median_num_iter = len(dataloaders["train"])

    # Train loop
    model.train()
    for epoch_id in tqdm(range(start_epoch, task_cfg.num_epochs), desc="Epoch"):
        for step in tqdm(range(median_num_iter), desc="Iters"):
            assert model.training
            iter_id = start_iter_id + step + (epoch_id * median_num_iter)

            loss, score, _, _ = forward_model(
                task_cfg, device, model, dataloaders, "train"
            )

            # Compute gradients
            loss.backward()
            clip_gradients(model, task_cfg["max_grad_norm"])

            # Apply and reset gradients
            optimizer.step()
            warmup_scheduler.step()
            model.zero_grad()

            # Increment loggers
            global_step += 1
            loss_values.append(loss)
            score_values.append(score)

            # Handle logging
            if step % 41 == 0 and step != 0:
                loss_avg, score_avg = float(sum(loss_values) / len(loss_values)), float(
                    sum(score_values) / len(score_values)
                )
                loss_values, score_values = [], []
                log_str = f"Epoch: {epoch_id}: Iter: {iter_id};  loss = {loss_avg:.3f}; accuracy  = {score_avg:.3f}"
                if step % 100 == 0:
                    log_str += f"\n lr rates = {[float(grp['lr']) for grp in optimizer.param_groups]}"
                logger.info(log_str)

        if epoch_id % 9 ==0:
            curr_val_score = evaluate(dataloaders, task_cfg, device, model)
            logger.info(
                f"[Validation] Current VQA: {curr_val_score:.3f} at {global_step} | Best VQA: {best_val_score:.3f} at {best_val_step}"
            )

        if curr_val_score > best_val_score:
            logger.info(f"Saving Checkpoint: {checkpoint_path}")
            model_to_save = model.module if hasattr(model, "module") else model
            best_val_score, best_val_step = curr_val_score, global_step
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "global_step": global_step,
                    "current_val_score": curr_val_score,
                    "epoch_id": epoch_id,
                },
                checkpoint_path,
            )

    print(
        f"Best Validation Score: {best_val_score:.3f}, Best Validation Epoch: {best_val_step}"
    )
    return checkpoint_path, model, dataloaders


def evaluate(dataloaders, task_cfg, device, model):
    scores, batch_sizes = [], []
    model.eval()
    with torch.no_grad():
        for batch_dict in tqdm(dataloaders["val"], desc="Validation"):
            loss, score, batch_size, _ = forward_model(
                task_cfg, device, model, batch_dict=batch_dict
            )
            scores.append(score * batch_size)
            batch_sizes.append(batch_size)

    model.train()
    return sum(scores) / sum(batch_sizes)


if __name__ == "__main__":
    checkpoint_path, model, dataloaders = main()

    assert os.path.exists(checkpoint_path)
    task = registry["val_on"][0]
    evaluator = Evaluator(checkpoint_path, model, dataloaders, task)

    # Beam search code has developed a problem and will be fixed in future!
    for beam_size in [1]:
        for split in ["test", "val"]:
            evaluator.evaluate_no_beam(split=split)
