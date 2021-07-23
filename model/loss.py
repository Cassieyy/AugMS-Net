import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
SMOOTH = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0) # batch_size
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=0.5, beta=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = SoftDiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)

class Active_Contour_Loss(nn.Module):
    def __init__(self):
        super(Active_Contour_Loss, self).__init__()

    def forward(self, y_true, y_pred): 
        
        x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] 
        y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

        delta_x = x[:,:,1:,:-2]**2
        delta_y = y[:,:,:-2,1:]**2
        delta_u = torch.abs(delta_x + delta_y) 

        lenth = torch.mean(torch.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((256, 256))
        C_2 = torch.zeros((256, 256))

        region_in = torch.abs(torch.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
        region_out = torch.abs(torch.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

        lambdaP = 1 # lambda parameter could be various.
        mu = 1 # mu parameter could be various.
        
        return lenth + lambdaP * (mu * region_in + region_out) 


# FL(pt) = −αt(1 − pt)^γ log(pt)
class FocalLoss(nn.Module):
    def __init__(self): 
        # assert 1>3 
        super(FocalLoss, self).__init__()
    # focalloss基于bce()
    def forward(self, classifications, targets):
        classifications = nn.Sigmoid()(classifications)
        alpha = 0.5 
        gamma = 2.0 
        classifications =  (classifications.squeeze(1)).squeeze(0) # DHW
        targets = (targets.squeeze(1)).squeeze(0) #DHW
        seqDepth = classifications.shape[0]
        classification_losses = []
        
        for j in range(seqDepth):
            classification = classifications[j, :, :] # HW
            target = targets[j, :, :]
            alpha_factor = torch.ones(target.shape).to(device) * alpha

            alpha_factor = torch.where(
                torch.eq(target, 1.), alpha_factor, 1. - alpha_factor)

            focal_weight = torch.where( 
                torch.eq(target, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma) #αt(1 − pt)γ
            bce = -(target * torch.log(classification + SMOOTH) +(1.0 - target) * torch.log(1.0 - classification + SMOOTH))
            cls_loss = focal_weight * bce
            classification_losses.append(cls_loss.mean())
        return torch.stack(classification_losses).mean(dim=0)

class DiceLoss(nn.Module): 
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        log_prob = torch.sigmoid(logits)                                                                                                                                            
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        return torch.mean(1. - dice_score)

class TverskyLoss(nn.Module):
    def __init__(self, alpha = 0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha

    def forward(self, y_true,y_pred): 
        y_true = (y_true.squeeze(1)).squeeze(0) 
        y_pred = (y_pred.squeeze(1)).squeeze(0) # DHW
        tp = (y_true * y_pred).sum() # 交集
        
        fp = ((1-y_true) * y_pred).sum()
        fn = (y_true * (1-y_pred)).sum()
        tversky = tp + SMOOTH / (tp + self.alpha*fp + self.beta*fn + SMOOTH) 

        tversky_loss = 1 - tversky
        return tversky_loss

class CrossEntropyLoss:
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._backend_loss = nn.CrossEntropyLoss(self.weight,
                                                 ignore_index=self.ignore_index,
                                                 reduction=self.reduction)

    def __call__(self, input, target, scale=[0.4, 1.]):
        '''
        :param input: [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :param scale: [...]
        :return: loss
        '''
        if isinstance(input, tuple) and (scale is not None):
            loss = 0
            for i, inp in enumerate(input):
                loss = loss + scale[i] * self._backend_loss(inp, target)
            return loss
        else:
            return self._backend_loss(input, target)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.001):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        depth = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        
        count_h = self._tensor_size(x[:,:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,:,1:])
        h_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/(batch_size + depth)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]

if __name__ == "__main__":
    loss = TVLoss()
    inputs = torch.randn(8,1,256,256)
    targets = torch.randn(8,1,256,256)
    # bce = loss(targets, inputs)
    tv_reg = TVLoss(inputs)
    print(bce)