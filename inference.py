"""
load saved model and infer M4C to draw z^dec * obj/ocr
"""
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
from sam.task_utils import (forward_model,
                            get_optim_scheduler, load_datasets)
from tools.registry import registry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

    parser.add_argument("--pretrained_eval", default="save/tvqa/best_model.pt", 
        help="Path of pretrained checkpoint")
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

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # When running only evaluation
    if opt.pretrained_eval != "":
        logger.info(
            f"Dumping Evaluation results at: {os.path.dirname(opt.pretrained_eval)}"
        )
        # add load model here
        model.load_state_dict(torch.load(opt.pretrained_eval)['model_state_dict'])

    curr_val_score = evaluate(dataloaders['val'], task_cfg, device, model)


def evaluate(dataloader, task_cfg, device, model):
    scores, batch_sizes = [], []
    model.eval()
    with torch.no_grad():
        for batch_dict in tqdm(dataloader, desc="Validation"):
            loss, score, batch_size, _ = forward_model(  # need a new forward
                task_cfg, device, model, batch_dict=batch_dict, evaluate=True
            )
            scores.append(score * batch_size)
            batch_sizes.append(batch_size)

    model.train()
    return sum(scores) / sum(batch_sizes)


if __name__ == "__main__":
    main()

    assert os.path.exists(checkpoint_path)
    task = registry["val_on"][0]
    evaluator = Evaluator(checkpoint_path, model, dataloaders, task)

    # Beam search code has developed a problem and will be fixed in future!
    for beam_size in [1]:
        for split in ["test", "val"]:
            evaluator.evaluate_no_beam(split=split)
