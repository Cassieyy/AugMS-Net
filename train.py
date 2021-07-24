import os
import time
import sys
sys.path.append("./model/")
sys.path.append("./model/Unet3D/")
sys.path.append("./model/Vnet/")
sys.path.append("./model/v2model")
sys.path.append("./model/AMEAnet/")
sys.path.append("./model/attentionUnet3D")
sys.path.append("./model/AMEA_deepvision")
sys.path.append("./model/AMEA_res2block")
sys.path.append("./model/PEUnet")
sys.path.append("./model/AMEA_deepvision_res2block")
from Unet3D import Unet3D
from AMEA_deepvision import AMEA_deepvision
from AMEA_deepvision_res2block import AMEA_deepvision_res2block
from AMEA_res2block import AMEA_res2block
from Vnet import VNet  
from attentionUnet3D import scSEUnet3D
from PEUnet import PEUnet3D
from U2Net3D import U2NET
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.oneSeq import oneSeq
from model.loss import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)     
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(model, criterion, optimizer, device, dataload, num_epochs=22,
                                save_epoch = 1, log_name = './train_log', checkpoint_name='/home/data/lpyWeight/paper/experiment',
                                loss_fre=2,img_fre=20,record = False):
    
    if record:
        log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)
        save_dirname =  os.path.join(checkpoint_name, dataset_model_lossfunc)
        if not os.path.exists(save_dirname):
            os.makedirs(save_dirname)
        writer = SummaryWriter(log_dirname)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_size = dataload.batch_size
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        
        for i, data in tqdm(enumerate(dataload),total=len(dataset)): 
            step += 1
            input = data[0]
            target = data[1]
            inputs = input.float().cuda(non_blocking=True)
            target = target.float().cuda(non_blocking=True)
            if deepvision:
                target1 = F.interpolate(target, size=(target.shape[2], target.shape[3] // 2, target.shape[4] // 2))
                target2 = F.interpolate(target, size=(target.shape[2], target.shape[3] // 4, target.shape[4] // 4))

                outputs, output1, output2 = model(inputs)
                loss_output = 0.7 * criterion(outputs, target)
                loss_output1 = 0.2 * criterion(output1, target1) 
                loss_output2 = 0.1 * criterion(output2, target2)
                loss = loss_output + loss_output1 + loss_output2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, target)

            # torch.autograd.set_detect_anomaly(True) # for debug

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss = epoch_loss + loss.item()
            print("{}/{},train_loss:{:.2}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))
            if record:
                if i % loss_fre == 0:
                    if not deepvision:
                        writer.add_scalar('train/loss:', loss.item(), epoch*dt_size+batch_size*i)
                    if deepvision:
                        writer.add_scalars('total&loss_o&loss_o1&loss_o2/loss:', {'total':loss.item(), 'loss_o':loss_output.item(), 'loss_o1':loss_output1.item(), 'loss_o2':loss_output2.item()}, epoch*dt_size+batch_size*i)
        lr_scheduler.step()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch % save_epoch==0:
            torch.save(model.state_dict(), os.path.join(save_dirname,'epoch_{}.pth'.format(epoch)))

def train(model,criterion,optimizer,device,dataloader,record,num_epochs):
    train_model(model, criterion, optimizer, device,dataloader,record=record,num_epochs=num_epochs)

def load_model_checkpoints(model,checkpoint_path='/home/data/lpyWeight/paper/experiment/oneSeq_AMnet_fl_n4_2/epoch_63.pth'):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    begin_time = time.time()
    batch_size = 1
    deepvision = False
    focal_loss = True    
    record = True
    lr = 0.001 
    num_epochs = 200
    continue_train = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset_model_lossfunc = "oneSeq_Unet3D_fl"
    # dataset_model_lossfunc = "oneSeq_AMnet_fl"
    # dataset_model_lossfunc = "oneSeq_Vnet_fl_1"
    # dataset_model_lossfunc = "oneSeq_scSEUnet3D_fl_1"
    # dataset_model_lossfunc = "oneSeq_PEUnet3D_fl_1"
    # dataset_model_lossfunc = "oneSeq_U2net3D_fl_1"
    dataset_model_lossfunc = "oneSeq_res2block_fl_1"

    # model = Unet3D(1, 1).to(device)
    model = AMEA_res2block(1, 1).to(device)
    # model = AMEA_deepvision_res2block(1, 1, scale=4).to(device)
    # model = scSEUnet3D(1, 1).to(device)
    # model = VNet().to(device)
    # model = AMEA_deepvision(1, 1).to(device)
    # model = PEUnet3D(1, 1).to(device)
    # model = U2NET(1, 1).to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    dataset = oneSeq(root="/home/data/redhouse", train=True)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    print('*' * 15,'batch_sizes = {}'.format(batch_size),'*' * 15)
    print('*' * 15,'lr = {}'.format(lr),'*' * 15)
    print('*' * 15,'criterion?focal_loss = {}'.format(focal_loss),'*' * 15)
    print('*' * 15,'device {}'.format(os.environ['CUDA_VISIBLE_DEVICES']),'*' * 15)
    if continue_train:
        load_checkpoint_path = '/home/data/lpyWeight/paper/experiment/oneSeq_AMnet_fl_n4_3/epoch_127.pth'
        print('*' * 15,'continue training...','*' * 15)
        load_model_checkpoints(model,load_checkpoint_path)
    if focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    train(model,criterion,optimizer,device,dataloader,record,num_epochs)
    end_time = time.time()
    run_time = end_time - begin_time
    print ('total running time：',run_time) 
