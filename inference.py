"""
load saved model and infer M4C to draw z^dec * obj/ocr
"""
import argparse
import logging
import os
import sys
import random
import json
import urllib
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
    obj_json_list, ocr_json_list = [], []
    with open(f'/home/qiyuan/2022spring/sam-textvqa/data/textvqa/TextVQA_0.5.1_val.json', 'r') as f:
        json_content = f.read()
        obj_json_list = json.loads(json_content)['data']
        obj_qids = [item['question_id'] for item in obj_json_list]
        
    with open(f'/home/qiyuan/2022spring/sam-textvqa/data/textvqa/TextVQA_Rosetta_OCR_v0.2_val.json', 'r') as f:
        json_content = f.read()
        ocr_json_list = json.loads(json_content)['data']

    scores, batch_sizes = [], []

    model.eval()
    with torch.no_grad():
        for i, batch_dict in tqdm(enumerate(dataloader), desc="Validation"):
            if i < 100:
                qid = batch_dict['question_id'].item()
                json_idx = obj_qids.index(qid)
                obj_json = obj_json_list[json_idx]
                ocr_json = ocr_json_list[json_idx]
                batch_dict['obj_json'] = obj_json
                batch_dict['ocr_json'] = ocr_json
                loss, score, batch_size, _ = forward_model(
                    task_cfg, device, model, batch_dict=batch_dict, infer=True
                )
                scores.append(score * batch_size)
                batch_sizes.append(batch_size)

    # model.train()
    return sum(scores) / sum(batch_sizes)


def download_val_images():
    data_folder = '/home/qiyuan/2022spring/sam-textvqa/data/textvqa/'
    with open(f'{data_folder}TextVQA_0.5.1_val.json', 'r') as f:
        json_content = f.read()
        obj_json_list = json.loads(json_content)['data']
        img_urls = [item['flickr_300k_url'] for item in obj_json_list]
    
    for i, img_url in enumerate(img_urls):
        img_name = obj_json_list[i]['image_id']
        save_imgfile = f'/home/qiyuan/2022spring/sam-textvqa/data/textvqa/val_images/{img_name}.jpg'
        with open(save_imgfile, 'wb') as f2:
            try:
                f2.write(urllib.request.urlopen(img_url).read())
            except:
                print(f'img {img_name} not found')
                os.remove(save_imgfile)
            f2.close()
            print(f'download {img_name} successful')

    files = os.listdir(f'{data_folder}/val_images')  # downloaded 2859 imgs
    print(len(files))


if __name__ == "__main__":
    # download_val_images()  # run once
    main()

    assert os.path.exists(checkpoint_path)
    task = registry["val_on"][0]
    evaluator = Evaluator(checkpoint_path, model, dataloaders, task)

    # Beam search code has developed a problem and will be fixed in future!
    for beam_size in [1]:
        for split in ["test", "val"]:
            evaluator.evaluate_no_beam(split=split)
