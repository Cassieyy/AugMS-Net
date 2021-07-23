import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class oneSeq(Dataset):
    def __init__(self,root="/home/share/RedHouse/lpyPatientpredict",img_size=256,img_transform=None, mask_transform=None, train=False):
        self.root = root
        self.img_size = img_size
        self.train = train
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.imgs = self.checkCountNum(root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        data = self.imgs[index]
        img = data[0].unsqueeze(0) 
        mask = data[1].unsqueeze(0)
        label = data[2].unsqueeze(0)
        save_path = data[3]
        h_size = data[4]
        w_size = data[5]
        return(img, mask, label, save_path, h_size, w_size)
    #[C,H,W]
    def crop_center(self, img, cropx, cropy):
        _,x,y = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        return img[:,starty:starty+cropy, startx:startx+cropx]

    def clamp(self, niiImage):
        y1 = np.percentile(niiImage, 0.2)
        y2 = np.percentile(niiImage, 99.8)    
        niiImage[niiImage < y1] = y1
        niiImage[niiImage > y2] = y2
        return niiImage

    def masklabel(self, mask):
        c, _, _ = mask.shape
        label = torch.zeros(c, 1)
        for index in range(c):
            if torch.sum(mask[index, :, :]) == 0:
                label[index] = 0
            else:
                label[index] = 1
        return label

    def checkCountNum(self, root, which_indexs = ['2']):
        # CountNum = []
        standardForm = [] 
        if self.train:
            for fir_dir in ['DONE0705', 'DONE0712']:                  
                for sec_dir in os.listdir(os.path.join(root,fir_dir)): #['钱慧君-79', '张娣-26', '凌蕾-74', '陆沈明-28', '贺蓓仪-23', '王晶-45', '谢洁林-50', '钱岚-51', '闵祺尔-48']
                    for thi_dir in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation')): # [05451715]
                        save_path = os.path.join(root, fir_dir, sec_dir, 'segmentation', thi_dir)
                        for tempSeq in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir)):
            
                            index = tempSeq.split(".")[-2]
                            tempmainSeq = ".".join(tempSeq.split(".")[:-2])+"."+(tempSeq.split(".")[-1])
                            
                            mainPath = os.path.join(root, fir_dir,  sec_dir, 'main',thi_dir, tempmainSeq)
                            segPath = os.path.join(root, fir_dir, sec_dir, 'segmentation', thi_dir, tempSeq)
                            if not (os.path.exists(mainPath)):
                                continue
                            if index in which_indexs:
                                temp_img_mask_label = []
                                main_data = nib.load(mainPath)
                                seg_data = nib.load(segPath)
                                main_data = self.clamp(main_data.get_fdata())
                                seg_data = seg_data.get_fdata()

                                h_size = torch.Tensor(main_data).permute(2, 0, 1).shape[1]
                                w_size = torch.Tensor(main_data).permute(2, 0, 1).shape[2]
                            
                                tensor_maindata = self.crop_center(torch.Tensor(main_data).permute(2, 0, 1), self.img_size, self.img_size)
                                tensor_segdata = self.crop_center(torch.Tensor(seg_data).permute(2, 0, 1), self.img_size, self.img_size)

                                standardlabel = self.masklabel(tensor_segdata) 
                                if not((tensor_maindata.shape)[0] == (tensor_segdata.shape)[0] == standardlabel.shape[0]):
                                    print((tensor_maindata.shape)[0], (tensor_segdata.shape)[0], standardlabel.shape[0])
                                    assert 1>3
                                # CountNum.append(tempNum)
                                temp_img_mask_label.append(tensor_maindata)
                                temp_img_mask_label.append(tensor_segdata)
                                temp_img_mask_label.append(standardlabel)
                                temp_img_mask_label.append(save_path + '/' + tempSeq)
                                temp_img_mask_label.append(h_size)
                                temp_img_mask_label.append(w_size)
                                standardForm.append(temp_img_mask_label)
                                # CountNum.append(tempNum)
            return standardForm

        else:
            standardForm = []
            for sec_dir in os.listdir(os.path.join(root, 'The_third')): #['111', '113'...]
                for thi_dir in os.listdir(os.path.join('/home/data/redhouse/The_third', sec_dir, 'segmentation')):
                    save_path = os.path.join(root, 'The_third',  sec_dir, 'segmentation', thi_dir)
                    for tempSeq in os.listdir(os.path.join('/home/data/redhouse/The_third', sec_dir, 'segmentation', thi_dir)):       
                        tempmainSeq = ".".join(tempSeq.split(".")[:-2])+"."+(tempSeq.split(".")[-1])
                        index = tempSeq.split('.')[-2]
                        mainPath = os.path.join(root, 'The_third',  sec_dir, 'main',thi_dir, tempmainSeq)
                        segPath = os.path.join(root, 'The_third',  sec_dir, 'segmentation', thi_dir, tempSeq)
                        if not (os.path.exists(mainPath)):
                            continue
                        if index in which_indexs:
                            temp_img_mask_label = []
                            main_data = nib.load(mainPath)
                            seg_data = nib.load(segPath)
            
                            main_data = self.clamp(main_data.get_fdata())
                            seg_data = seg_data.get_fdata()

                            h_size = torch.Tensor(main_data).permute(2, 0, 1).shape[1]
                            w_size = torch.Tensor(main_data).permute(2, 0, 1).shape[2]
                            
                            tensor_maindata = self.crop_center(torch.Tensor(main_data).permute(2, 0, 1), self.img_size, self.img_size)
                            tensor_segdata = self.crop_center(torch.Tensor(seg_data).permute(2, 0, 1), self.img_size, self.img_size)
                            
                            standardlabel = self.masklabel(tensor_segdata) 
                            if not((tensor_maindata.shape)[0] == (tensor_segdata.shape)[0] == standardlabel.shape[0]):
                                print(mainPath)
                                print((tensor_maindata.shape)[0], (tensor_segdata.shape)[0], standardlabel.shape[0])
                                assert 1>3
                            # CountNum.append(tempNum)
                            temp_img_mask_label.append(tensor_maindata)
                            temp_img_mask_label.append(tensor_segdata)
                            temp_img_mask_label.append(standardlabel)
                            temp_img_mask_label.append(save_path + '/' + tempSeq)
                            temp_img_mask_label.append(h_size)
                            temp_img_mask_label.append(w_size)
                            standardForm.append(temp_img_mask_label)
                            
            return standardForm

if __name__ == "__main__":
    batch_size = 1
    root = "/home/data/redhouse/"
    dataset = oneSeq(root, train = False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i,(img, mask, label, save_path, h_size, w_size) in enumerate(dataloader):
        print(i, h_size, w_size)
        assert 1>3
    