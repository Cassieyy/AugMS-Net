import cv2
import numpy as np
import torch
from hausdorff import hausdorff_distance
np.seterr(divide='ignore', invalid='ignore')
SMOOTH = 1e-5

'''
    希望得到的指标结果：
    1.Dice Similarity Coeffcient ✅
    2.IoU 要用混淆矩阵
    3.Sensitivity/Recall✅
    4.ppv/cpa/Precision✅
    5.Hausdorff_95(95%HD) 单位mm dice对mask内部填充比较敏感,而hausdorff distance对分割边界敏感✅
    6.Accuracy(准确率)✅
    7.para 在test中✅
'''
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes): #我们应该是2
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes)) # matrix 2*2
 
    def _fast_hist(self, label_pred, label_true): # 计算一行(1*256)的混淆矩阵
        # 找出标签中需要计算的类别 去掉背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes) ## core code
        return hist
 
    def add_batch(self, predictions, gts):# 计算一张256*256图的混淆矩阵 print[[65536. 0.][0. 0.]]
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())# flatten()按照行展成一行
    def evaluate(self): # 对单张图像
        np.seterr(divide='ignore',invalid='ignore')
        # 6.Accuracy
        Accuracy = (np.diag(self.hist).sum() + SMOOTH) / (self.hist.sum() + SMOOTH) # PA = 识别正确的像素/全部像素  精确率
        # acc_cls = np.diag(self.hist) / self.hist.sum(axis=0)  # cpa 横是真实 纵是预测  类别精确率
        
        # 4.ppv/cpa/Precision 返回两个值 neg pos
        # cpa = np.diag(self.hist) / (self.hist.sum(axis = 0) + SMOOTH) # 精准率 即cpa precision
        
        # 3.Sensitivity/Recall
        # Recall = np.diag(self.hist) / (self.hist.sum(axis = 1)+ SMOOTH) #召回率/灵敏度sensivity
    
        # 返回的是一个列表值，如：[0.90, 0.80]，表示类别1 2各类别的预测准确率
        # print("acc_cls", acc_cls)
        # acc_cls = np.nanmean(acc_cls)# nanmean()计算时分母不会加nanmean()的项数

        # 1.DSC
        # dsc = 2*(np.diag(self.hist)[1]) / (2*(np.diag(self.hist)[1]) + np.diag(np.fliplr(self.hist)).sum())
        Positive = 0
        # 统计正样本个数
        if self.hist.sum(axis=1)[1] != 0: # 统计真实值为正样本且不为1的个数
            Positive = 1
        #TrueFalseNum
        WrongNum = 0
        if np.diag(self.hist).sum() == 0:
            WrongNum = 1
        
        #IoU 这里是两个类的 就本项目来说 IoU得到的是两个类的iou
        IoU = (np.diag(self.hist) + SMOOTH) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + SMOOTH)  #IoU 并集SA + SB - (SA交SB)     
        
        #MIoU = sum(IoU)/self.num_classes
        mIoU = np.nanmean(IoU)
    
        # return acc, recallRate, cpa, Positive, WrongNum, mIoU
        return Accuracy, mIoU, Positive, WrongNum
 
 
 # mask [channel,h,w]
def get_iou(mask,predict,thr): # 得到一张的IoU
    # mask = mask.squeeze(0)
    # predict = predict.squeeze(0)
    assert mask.shape == predict.shape
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(depth, height, weight)
    # assert 1>3

    predict[predict < thr] = 0 
    predict[predict >= thr] = 1
    mask[mask < thr] = 0
    mask[mask >= thr] = 1
    # print(torch.equal(mask, predict))
    # mask[:8,:] = 1
    # print(mask.shape)
    predict = predict.numpy().astype(np.int16)
    mask = mask.numpy().astype(np.int16)

    Iou = IOUMetric(2)
    Iou.add_batch(predict, mask) # predict & mask都是256*256的 add_batch逐行判断
    
    # acc, recallRate, cpa, positive, wrongNum, miou= Iou.evaluate()
    Accuracy, mIoU, Positive, WrongNum = Iou.evaluate()
    return Accuracy, mIoU, Positive, WrongNum


# mask [batchsize,channel,h,w]
def get_miou(mask,predict,thr):# 得到一个batch的miou
    batchsize = mask.shape[0]
    m_iou = 0
    PositiveNum = 0
    WrongNum = 0
    acc = 0
    for i in range(batchsize):
        Accuracy, mIoU, Positive, WrongNum = get_iou(mask[i],predict[i],thr)
        PositiveNum += Positive
        WrongNum += WrongNum
        m_iou += mIoU
        acc += Accuracy 
    return m_iou / batchsize, acc / batchsize, PositiveNum, WrongNum

def dice_coef(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + SMOOTH) / \
        (output.sum() + target.sum() + SMOOTH)

def accuracy(output, target):
    # # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # # torch才有view这个函数
    # # batchsize = output.shape[0]
    # # for i in range(batchsize):
    # output = output.view(-1).data.cpu().numpy()
    # output = (np.round(output)).astype('int') # 相当于阈值分隔
    # target = target.view(-1).data.cpu().numpy()
    # target = (np.round(target)).astype('int')
    # (output == target).sum()
    # print(((output == target).sum()) / len(output.flatten()))
    # print(output.shape, target.shape)
    # print(len(output))
    # assert 1>3
    return (output == target).sum() / len(output.flatten())

def ppv(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + SMOOTH) / \
           (output.sum() + SMOOTH)

def sensitivity(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + SMOOTH) / \
        (target.sum() + SMOOTH)

def m_metric(mask, predict, thr):
    batchsize = mask.shape[0]
    
    mask = mask.squeeze(1)
    predict = predict.squeeze(1)
    
    predict[predict < thr] = 0 
    predict[predict >= thr] = 1
    mask[mask < thr] = 0
    mask[mask >= thr] = 1
    
    predict = predict.numpy().astype(np.int16)
    mask = mask.numpy().astype(np.int16)

    m_dsc = []
    m_acc = []
    m_ppv = []
    m_sen = []
    m_hausdorff_distance = []

    for i in range(batchsize):
        m_dsc.append(dice_coef(predict[i], mask[i]))
        m_acc.append(accuracy(predict[i], mask[i]))
        m_ppv.append(ppv(predict[i], mask[i]))
        m_sen.append(sensitivity(predict[i], mask[i]))
        m_hausdorff_distance.append(hausdorff_distance(predict[i], mask[i]))

    return np.nanmean(m_dsc), np.nanmean(m_acc), np.nanmean(m_ppv), np.nanmean(m_sen), np.nanmean(m_hausdorff_distance)

if __name__ == "__main__":
    output = torch.randn(8, 1, 256, 256)
    target = torch.randn(8, 1, 256, 256)

    # Iou = IOUMetric(2)
    # Iou.add_batch(output.numpy(), target.numpy()) # predict & mask都是256*256的 add_batch逐行判断
    # iou = get_miou(target, output, 0.4)
    # print(iou)

    dsc, acc, ppv, sen, hf = m_metric(target, output, 0.4)
    print(dsc, acc, ppv, sen, hf)