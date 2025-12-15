from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.whubuilding_dataset import *
from tools.metric import Evaluator
from catalyst.contrib.nn import Lookahead
from catalyst import utils

task_name = 'UAGLNet_WHU'
root_path = './results'
save_path = os.path.join(root_path, task_name)
class_num = 2
evaluator = Evaluator(num_class=class_num)
n_epochs = 105
save_freq = 15
dev_freq = 1
batch_size = 8
train_batch_size = batch_size
test_batch_size = batch_size
val_batch_size = batch_size
backbone_lr = 5e-4
backbone_weight_decay = 0.0025
lr = 5e-4
weight_decay = 0.0025

from geoseg.models.UAGLNet import UAGLNet
net = UAGLNet(drop_path_rate=0.1)
loss_func = UAGLloss()

# define the dataloader
data_path = 'path/to/WHU'
train_dataset = WHUBuildingDataset(data_root=os.path.join(data_path,'train'), mode='train', mosaic_ratio=0.25, transform=train_aug)
val_dataset = WHUBuildingDataset(data_root=os.path.join(data_path,'val'), mode='val', transform=val_aug)
test_dataset = WHUBuildingDataset(data_root=os.path.join(data_path,'test'), mode='val', transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=train_batch_size,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=val_batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=test_batch_size,
                        num_workers=test_batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

layerwise_params = {"CoE.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
