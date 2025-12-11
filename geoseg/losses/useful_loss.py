import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss

class UAGLloss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=10.0):
        super(UAGLloss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor
        self.criterion = nn.BCELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0, max=1.0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def compute_uncertainty_loss(self, prob_x, targets):
        new_targets = targets.unsqueeze(1).float()
        loss_ug1 = 0.5*self.criterion(prob_x, new_targets) 
        loss_ug2 = 0.1*self.kl_loss(prob_x.clamp(0.0001, 0.9999).log(), new_targets.clamp(0.0001,0.9999))
        return loss_ug1 + loss_ug2

    def forward(self, logits, prob_h, prob_l, targets):
        loss = self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor
        loss_ug_g = self.compute_uncertainty_loss(prob_h, targets)
        loss_ug_l = self.compute_uncertainty_loss(prob_l, targets)
        
        final_loss = loss + loss_ug_g + loss_ug_l
        # print(loss, loss_ug_l, loss_ug_g)
        return final_loss
