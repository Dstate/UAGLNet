import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ttach as tta
import argparse
import numpy as np
import torch
from tools.cfg import py2cfg
from geoseg.models.UAGLNet import UAGLNet
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-d", "--dataset", type=str, default="Inria", choices=["Inria", "Mass", "WHU"], help="Dataset to use: inria, mass, or WHU")
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr", "xxx"])
    return parser.parse_args()

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
    return pred_results, img_ids

def main():
    args = get_args()

    if args.dataset == "Inria":
        config_path = "config/inria/UAGLNet.py"
        ckpt_path = "ldxxx/UAGLNet_Inria"
    elif args.dataset == "WHU":
        config_path = "config/WHU/UAGLNet.py"
        ckpt_path = "ldxxx/UAGLNet_WHU"
    elif args.dataset == "Mass":
        config_path = "config/mass/UAGLNet.py"
        ckpt_path = "ldxxx/UAGLNet_Massachusetts"
    
    config = py2cfg(config_path)    
    test_loader = config.test_loader
    evaluator = config.evaluator
    model = UAGLNet.from_pretrained(ckpt_path)
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


if __name__ == "__main__":
    main()