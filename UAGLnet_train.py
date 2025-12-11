import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
from tools.cfg import py2cfg
import torch
import numpy as np
import argparse
from pathlib import Path
import random
from tqdm import tqdm
import copy

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print("current seed is ",seed)

seed_everything(42)

def model_save(model, save_path):
    try :
        state_dict = model.module.state_dict()
    except:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)

def dev(model, val_loader, evaluator):
    for batch in tqdm(val_loader, position=0):
        img, mask = batch['img'], batch['gt_semantic_seg']
        img, mask = img.cuda(), mask.cuda()
        with torch.no_grad():
            pred = model(img)
            pre_mask = torch.nn.Softmax(dim=1)(pred)
            pre_mask = pre_mask.argmax(dim=1)
            for i in range(mask.shape[0]):
                evaluator.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

    return evaluator.Intersection_over_Union()

def train(config, model, train_loader, val_loader, loss_func, optimizer, scheduler, evaluator):
    epoch = 0
    best_ep = -1
    best_record = 0.
    n_epoch = config.n_epochs
    save_freq = config.save_freq
    dev_freq = config.dev_freq
    train_eval = copy.deepcopy(evaluator)
    while epoch < n_epoch:
        epoch += 1
        model.train()
        loss_records = []
        train_eval.reset()

        print('  Training Epoch  %3d:' % epoch)
        for batch in tqdm(train_loader, position=0):
            img, mask = batch['img'], batch['gt_semantic_seg']
            img, mask = img.cuda(), mask.cuda()
            
            pred, prob_high, prob_low = model(img)
            pre_mask = torch.nn.Softmax(dim=1)(pred)
            pre_mask = pre_mask.argmax(dim=1)
            for i in range(mask.shape[0]):
                train_eval.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

            loss = loss_func(pred, prob_high, prob_low, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_records.append(loss)
        
        scheduler.step()
        logging.info('    Training (Epoch %3d): loss = %.4f    ' % (epoch, sum(loss_records)/len(loss_records)))
        
        if epoch % dev_freq == 0 :
            model.eval()
            evaluator.reset()
            print('  Dev Epoch  %3d:' % epoch)
            dev_records = dev(model, val_loader, evaluator)
            logging.info('    Dev result (Epoch %4d): train_mIoU = %s val_mIoU =  %s    ' % (epoch, str(train_eval.Intersection_over_Union()), str(dev_records)))
           
            if best_record < dev_records[0]:
                best_ep = epoch
                best_record = dev_records[0]
                model_save(model, os.path.join(config.save_path, 'ckpts', 'best_epoch.pth'))
                
        if epoch % save_freq == 0:
            model_save(model, os.path.join(config.save_path, 'ckpts', str(epoch) + '.pth'))

    logging.info('    Best epoch is (Epoch %4d): val_mIoU = %s    ' % (best_ep, str(best_record)) )

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default="config/inria/UGLnet.py")
    return parser.parse_args()

def create_save_path(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, 'visualization')):
        os.mkdir(os.path.join(save_path, 'visualization'))
    if not os.path.exists(os.path.join(save_path, 'ckpts')):
        os.mkdir(os.path.join(save_path, 'ckpts'))

def main():
    print("train process begins")
    args = get_args()
    config = py2cfg(args.config_path)
    create_save_path(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path,'log.txt'),
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.INFO)

    train_loader = config.train_loader
    val_loader = config.val_loader
    loss_func = config.loss_func
    evaluator = config.evaluator
    model = config.net
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    optimizer = config.optimizer
    scheduler = config.lr_scheduler

    train(config, model, train_loader, val_loader, loss_func, optimizer, scheduler, evaluator)


if __name__ == "__main__":
   main()
