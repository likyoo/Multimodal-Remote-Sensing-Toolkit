import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Cross_fusion_CNN_Loss(nn.Module):
    # Loss function for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    def __init__(self, weight):
        super(Cross_fusion_CNN_Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, output, target):
        output1, output2, output3 = output
        loss1 = self.ce(output1, target)
        loss2 = torch.pow(output1 - output2, 2).mean()
        loss3 = torch.pow(output1 - output3, 2).mean()

        return loss1 + loss2 + loss3

class EndNet_Loss(nn.Module):
    # loss function for "Deep Encoder–Decoder Networks for Classification of Hyperspectral and LiDAR Data"
    def __init__(self, weight):
        super(EndNet_Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, output, target):
        out, de_x1, de_x2, ori_x1, ori_x2 = output  # de_x1 means the output of decoder, ori means original data
        loss1 = self.ce(out, target)
        loss2 = self.mse1(de_x1, ori_x1)
        loss3 = self.mse1(de_x2, ori_x2)

        return loss1 + loss2 + loss3


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()