'''
Re-implementation for paper "FusAtNet: Dual Attention based SpectroSpatial Multimodal Fusion Network for
Hyperspectral and LiDAR Classification"
The official keras implementation is in https://github.com/ShivamP1993/FusAtNet
'''

import torch.nn as nn
import torch

class ConvUnit(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x

class ConvUnit_NP(nn.Module):
    # No Padding
    def __init__(self, input_channels, output_channels):
        super(ConvUnit_NP, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x

class Residual_Unit1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Residual_Unit1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        identity = x
        x = self.activation(self.bn2(self.conv2(x)))
        x += identity
        x = self.max_pool(x)
        return x

class Residual_Unit2(nn.Module):
    # without pooling
    def __init__(self, input_channels, output_channels):
        super(Residual_Unit2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        identity = x
        x = self.activation(self.bn2(self.conv2(x)))
        x += identity
        return x

class Hyper_Feature_Extractor(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super(Hyper_Feature_Extractor, self).__init__()
        self.conv1 = ConvUnit(input_channels, 256)
        self.conv2 = ConvUnit(256, 256)
        self.conv3 = ConvUnit(256, 256)

        self.conv4 = ConvUnit(256, 256)
        self.conv5 = ConvUnit(256, 256)
        self.conv6 = ConvUnit(256, output_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out

class Spectral_Attention_Module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super(Spectral_Attention_Module, self).__init__()
        self.res1 = Residual_Unit1(input_channels, 256)
        self.res2 = Residual_Unit1(256, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.avg_pool(out)
        return out

class Spatial_Attention_module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super(Spatial_Attention_module, self).__init__()
        self.res1 = Residual_Unit2(input_channels, 128)
        self.res2 = Residual_Unit2(128, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class Modality_Feature_Extractor(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super(Modality_Feature_Extractor, self).__init__()
        self.conv1 = ConvUnit(input_channels, 256)
        self.conv2 = ConvUnit(256, 256)
        self.conv3 = ConvUnit(256, 256)

        self.conv4 = ConvUnit(256, 256)
        self.conv5 = ConvUnit(256, 256)
        self.conv6 = ConvUnit(256, output_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out

class Modality_Attention_Module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super(Modality_Attention_Module, self).__init__()
        self.res1 = Residual_Unit2(input_channels, 128)
        self.res2 = Residual_Unit2(128, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class Classification_Module(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Classification_Module, self).__init__()
        self.conv1 = ConvUnit_NP(input_channels, 256)
        self.conv2 = ConvUnit_NP(256, 256)
        self.conv3 = ConvUnit_NP(256, 256)

        self.conv4 = ConvUnit_NP(256, 256)
        self.conv5 = ConvUnit_NP(256, 1024)
        self.conv6 = nn.Conv2d(1024, num_classes, kernel_size=1, bias=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = torch.squeeze(out)
        return out

class FusAtNet(nn.Module):
    def __init__(self, input_channels, input_channels2, num_classes):
        super(FusAtNet, self).__init__()
        self.hfe = Hyper_Feature_Extractor(input_channels, 1024)
        self.spectral_am = Spectral_Attention_Module(input_channels, 1024)
        self.spatial_am = Spatial_Attention_module(input_channels2, 1024)
        self.mfe = Modality_Feature_Extractor(1024 * 2 + input_channels + input_channels2, 1024)
        self.mam = Modality_Attention_Module(1024 * 2 + input_channels + input_channels2, 1024)
        self.cm = Classification_Module(1024, num_classes)

    def forward(self, x1, x2):
        Fhs = self.hfe(x1)
        Ms = self.spectral_am(x1) * Fhs
        Mt = self.spatial_am(x2) * Fhs
        Fm =self.mfe(torch.cat([x1, x2, Ms, Mt], 1))
        Am = self.mam(torch.cat([x1, x2, Ms, Mt], 1))
        Fss = Fm * Am
        out = self.cm(Fss)
        return out


