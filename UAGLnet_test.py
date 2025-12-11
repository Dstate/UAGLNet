import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tools.cfg import py2cfg

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default="/home/ldx/Reproduction/UGLnet/config/inria/UGLnet_f3.py", help="Path to the config")
    # arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr", "xxx"])
    # arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()

def model_load(model, load_path):
    test_epoch = 'best_epoch.pth'
    load_path = os.path.join(load_path, test_epoch)
    logging.info('    Test epoch is %s    ' % (test_epoch))
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    print("load from", load_path)
    return model

def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb

def print_mask(pred_results, img_id, save_path):
    length = len(img_id)
    print('print mask process:')
    for i in tqdm(range(length), position=0):
        mask = label_to_rgb(pred_results[i])
        save_name = os.path.join(save_path,str(img_id[i])+'.png')
        cv2.imwrite(save_name, mask)

def test_pred(model, test_loader, evaluator):
    evaluator.reset()
    pred_results = []
    img_ids = []

    print('test process:')
    for batch in tqdm(test_loader, position=0):
        img, mask, img_id = batch['img'], batch['gt_semantic_seg'], batch["img_id"]
        img, mask = img.cuda(), mask.cuda()
        with torch.no_grad():
            pred = model(img)
            pre_mask = torch.nn.Softmax(dim=1)(pred)
            pre_mask = pre_mask.argmax(dim=1)
            for i in range(mask.shape[0]):
                evaluator.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
            pred_results.append(pre_mask)
            img_ids.append(img_id)
    
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    test_result = 'F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA,
                                                     np.nanmean(precision[:-1]), np.nanmean(recall[:-1]))
    print('test result is', test_result)
    logging.info('    Test Result: %s    ' % (test_result))
    return pred_results, img_ids

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    
    logging.basicConfig(filename=os.path.join(config.save_path,'log.txt'),
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.INFO)
    
    test_loader = config.test_loader
    evaluator = config.evaluator
    model = config.net
    model = model_load(model, os.path.join(config.save_path, 'ckpts'))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    
    pred_results, img_ids = test_pred(model, test_loader, evaluator)
    # print_mask(pred_results, img_ids, os.path.join(config.save_path, 'visualization'))

if __name__ == "__main__":
    main()