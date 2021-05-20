import torch
from torch import nn
from torch.nn import functional as F

class Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        """Implementation of SAEM: Spatial Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block if not specifed reduced to half
        """
        super(Spatial_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=size * size,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)
        t1 = t1.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*HW*TF --> B*TF*HW
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*HW
        Affinity_M = Affinity_M.view(batch_size, 1, x1.size(2), x1.size(3))   # B*1*H*W

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels, in_channels2, inter_channels=None, inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block
        """
        super(Spectral_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.in_channels2 = in_channels2
        self.inter_channels2 = inter_channels2

        if self.inter_channels is None:
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1
        if self.inter_channels2 is None:
            self.inter_channels2 = in_channels2
            if self.inter_channels2 == 0:
                self.inter_channels2 = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels2, out_channels=self.inter_channels2, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels2),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels2,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels2, -1)
        t2 = t2.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*C1*C2 --> B*C2*C1
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*C1
        Affinity_M = Affinity_M.view(batch_size, x1.size(1), 1, 1)  # B*C1*1*1

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out

class S2ENet(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes, patch_size):
        super(S2ENet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # For image a (7×7×input_channels) --> (7×7×planes_a[0])
        self.conv1_a = conv_bn_relu(input_channels, self.planes_a[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(input_channels2, self.planes_b[0], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[0]) --> (7×7×planes_a[1])
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[1], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[0], self.planes_b[1], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[1]) --> (7×7×planes_a[2])
        self.conv3_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[1]) --> (7×7×planes_b[2])
        self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        self.SAEM = Spatial_Enhance_Module(in_channels=self.planes_a[2], inter_channels=self.planes_a[2]//2, size=patch_size)
        self.SEEM = Spectral_Enhance_Module(in_channels=self.planes_b[2], in_channels2=self.planes_a[2])

        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.planes_a[2] * 2,
                out_channels=self.planes_a[2],
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.planes_a[2], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.conv1_a(x1)
        x2 = self.conv1_b(x2)

        x1 = self.conv2_a(x1)
        x2 = self.conv2_b(x2)

        x1 = self.conv3_a(x1)
        x2 = self.conv3_b(x2)

        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x2, x1)

        x = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    import torch

    img1 = torch.randn(2, 6, 7, 7)
    img2 = torch.randn(2, 6, 7, 7)

    SAEM = Spatial_Enhance_Module(in_channels=6, inter_channels=6 // 2, size=7)
    out = SAEM(img1, img2)
    print(out)

    SEEM = Spectral_Enhance_Module(in_channels=6, in_channels2=6)
    out = SEEM(img1, img2)
    print(out.shape)