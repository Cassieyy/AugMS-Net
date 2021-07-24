import os
import sys
sys.path.append("./model")
sys.path.append("./model/Vnet/")
sys.path.append("./model/Unet3D/")
sys.path.append("./model/v2model")
sys.path.append("./model/AMEAnet/")
sys.path.append("./model/attentionUnet3D")
sys.path.append("./model/AMEA_deepvision")
sys.path.append("./model/AMEA_res2block")
sys.path.append("./model/PEUnet")
sys.path.append("./model/AMEA_deepvision_res2block")
from AMEA_deepvision_res2block import AMEA_deepvision_res2block
from AMEA_res2block import AMEA_res2block   
from AMEA_deepvision import AMEA_deepvision
from attentionUnet3D import scSEUnet3D
from Vnet import VNet   
from Unet3D import Unet3D
from model.Unet3D_DM import Unet3D_DM
from U2Net3D import U2NET
from PEUnet import PEUnet3D
from AMEAnet import AMEAnet 
from dataset.oneSeq import oneSeq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import metric 
@torch.no_grad()
def test_model(model,device, dataload,thr):
    
    m_dsc = []
    m_acc = []
    m_ppv = []
    m_sen = []
    m_hausdorff_distance = []

    m_ious = []
    Accs = []
    Positive = 0
    Wrongnum = 0

    for _,data in enumerate(dataload):
        input = data[0]
        target = data[1]
        inputs = input.float().to(device)
        target = target.float().to(device)
        if deepvision:
            outputs = model(inputs)[0]
        else:
            outputs = model(inputs)
        
        for depth in range(outputs.shape[2]):
            m_iou, Acc, PositiveNum, WrongNum = metric.get_miou(target.cpu().detach()[:, :, depth, :, :],outputs.cpu()[:, :, depth, :, :].detach(),thr)
            dsc, acc, ppv, sen, hausdorff_distance = metric.m_metric(target.cpu().detach()[:, :, depth, :, :],outputs.cpu().detach()[:, :, depth, :, :],thr)
            m_dsc.append(dsc)
            m_acc.append(acc)
            m_ppv.append(ppv)
            m_sen.append(sen)
            m_hausdorff_distance.append(hausdorff_distance)

            m_ious.append(m_iou)
            Accs.append(Acc)
            Positive += PositiveNum
            Wrongnum += WrongNum
    return np.nanmean(m_ious), np.nanmean(Accs),np.nanmean(m_dsc), np.nanmean(m_acc), np.nanmean(m_ppv), np.nanmean(m_sen), np.nanmean(m_hausdorff_distance), Positive, Wrongnum

def test(model,device,dataloader,thr):
    return test_model(model,device,dataloader,thr)

def load_model_checkpoints(model,checkpoint_path='./checkpoints/newSeUnet/latest.pth'):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    
if __name__ == "__main__":
    batch_size = 1
    thr = 0.55
    test_datapath = "/home/data/redhouse/"
    deepvision = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset_model_lossfunc = "oneSeq_PEUnet_fl"
    # dataset_model_lossfunc = "oneSeq_scSEUnet3D_fl_1"
    # dataset_model_lossfunc = "oneSeq_PEUnet3D_fl_1"
    # dataset_model_lossfunc = "oneSeq_3DUnet_dm433"
    # dataset_model_lossfunc = "oneSeq_AMnet_fl_n4_dm721"
    # dataset_model_lossfunc = "oneSeq_AMnet_fl_n4_6"
    # dataset_model_lossfunc = "oneSeq_Unet3D_fl_1"
    dataset_model_lossfunc = "oneSeq_res2block_fl_1"
    
    # dataset_model_lossfunc = "oneSeq_AMnet_bdl"
    # dataset_model_lossfunc = "oneSeq_Vnet_fl_1"
    # dataset_model_lossfunc = "oneSeq_U2net3D_fl_1"
    # dataset_model_lossfunc = "oneSeq_AMnet_fl_n4_dm433"

    # model = Unet3D_DM(1, 1).cuda()
    # model = U2NET(1, 1).cuda()
    # model = AMEA_deepvision_res2block(1, 1, scale=4).cuda()
    # model = AMEAnet(1, 1, deepvision=True)
    # model = Unet3D(1, 1).to(device)
    # model = VNet().to(device)
    # model = scSEUnet3D(1, 1).to(device)
    # model = PEUnet3D(1, 1).to(device)
    # model = AMEA_deepvision(1, 1).to(device)
    model = AMEA_res2block(1, 1).to(device)
    tmp_checkpoint_path = os.path.join("/home/data/lpyWeight/paper/experiment/", dataset_model_lossfunc+'/')
    param_count = sum(param.numel() for param in model.parameters())
    model.eval()
    dataset = oneSeq(test_datapath, train = False)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    best_iou = 0
    best_epoch = -1
    print('*' * 15,'threshold:', thr, '*' * 15)

    for tmp in range(78):
        load_checkpoint_path = tmp_checkpoint_path + "epoch_{}.pth".format(tmp)
        print('*' * 15,'batch_sizes = {}'.format(batch_size),'*' * 15)
        load_model_checkpoints(model,load_checkpoint_path)
        m_ious, Accs, m_dsc, m_acc, m_ppv, m_sen, m_hausdorff_distance, Positive, Wrongnum = test(model,device,dataloader,thr)
        epoch_num = ((load_checkpoint_path.split('/'))[-1]).split('.')[0]
        params = sum(param.numel() for param in model.parameters()) / 1e6
        if m_ious > best_iou:
            best_iou = m_ious
            best_epoch = epoch_num
        print('#Params: %.1fM' % (params))
        print('*' * 15,epoch_num + ':','*' * 15)
        print('*' * 15,'param_count:', param_count, '*' * 15)
        print('*' * 15,"mIoU:", m_ious,'*' * 15)
        print('*' * 15,"Accuracy:", Accs,'*' * 15)

        print('*' * 15,"DiceScore:", m_dsc,'*' * 15)
        print('*' * 15,"m_acc:", m_acc,'*' * 15)
        print('*' * 15,"m_ppv:", m_ppv,'*' * 15)
        print('*' * 15,"m_sen:", m_sen,'*' * 15)
        print('*' * 15,"m_hausdorff_distance:", m_hausdorff_distance,'*' * 15)

    print(best_iou, best_epoch)
