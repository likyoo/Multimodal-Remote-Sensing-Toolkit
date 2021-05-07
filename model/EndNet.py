'''
Re-implementation for paper "Deep Encoder-Decoder Networks for Classification of Hyperspectral and LiDAR Data"
The official tensorflow implementation is in https://github.com/danfenghong/IEEE_GRSL_EndNet
'''

import torch.nn as nn
import torch

class EndNet(nn.Module):
    # Re-implemented middle_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(EndNet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Encoder
        # For image a (1×1×d)
        self.encoder_fc1_a = nn.Linear(input_channels, filters[0])
        self.encoder_bn1_a = nn.BatchNorm1d(filters[0])
        self.encoder_fc2_a = nn.Linear(filters[0], filters[1])
        self.encoder_bn2_a = nn.BatchNorm1d(filters[1])
        self.encoder_fc3_a = nn.Linear(filters[1], filters[2])
        self.encoder_bn3_a = nn.BatchNorm1d(filters[2])
        self.encoder_fc4_a = nn.Linear(filters[2], filters[3])
        self.encoder_bn4_a = nn.BatchNorm1d(filters[3])

        # For image b (1×1×d)
        self.encoder_fc1_b = nn.Linear(input_channels2, filters[0])
        self.encoder_bn1_b = nn.BatchNorm1d(filters[0])
        self.encoder_fc2_b = nn.Linear(filters[0], filters[1])
        self.encoder_bn2_b = nn.BatchNorm1d(filters[1])
        self.encoder_fc3_b = nn.Linear(filters[1], filters[2])
        self.encoder_bn3_b = nn.BatchNorm1d(filters[2])
        self.encoder_fc4_b = nn.Linear(filters[2], filters[3])
        self.encoder_bn4_b = nn.BatchNorm1d(filters[3])

        self.joint_encoder_fc5 = nn.Linear(filters[3] * 2, filters[3])
        self.joint_encoder_bn5 = nn.BatchNorm1d(filters[3])
        self.joint_encoder_fc6 = nn.Linear(filters[3], filters[2])
        self.joint_encoder_bn6 = nn.BatchNorm1d(filters[2])
        self.joint_encoder_fc7 = nn.Linear(filters[2], n_classes)
        self.joint_encoder_bn7 = nn.BatchNorm1d(n_classes)

        # Decoder
        self.decoder_fc1_a = nn.Linear(filters[3], filters[2])
        self.decoder_fc2_a = nn.Linear(filters[2], filters[1])
        self.decoder_fc3_a = nn.Linear(filters[1], filters[0])
        self.decoder_fc4_a = nn.Linear(filters[0], input_channels)

        self.decoder_fc1_b = nn.Linear(filters[3], filters[2])
        self.decoder_fc2_b = nn.Linear(filters[2], filters[1])
        self.decoder_fc3_b = nn.Linear(filters[1], filters[0])
        self.decoder_fc4_b = nn.Linear(filters[0], input_channels2)

    def forward(self, ori_x1, ori_x2):

        x1 = self.activation(self.encoder_bn1_a(self.encoder_fc1_a(ori_x1)))
        x2 = self.activation(self.encoder_bn1_b(self.encoder_fc1_b(ori_x2)))

        x1 = self.activation(self.encoder_bn2_a(self.encoder_fc2_a(x1)))
        x2 = self.activation(self.encoder_bn2_b(self.encoder_fc2_b(x2)))

        x1 = self.activation(self.encoder_bn3_a(self.encoder_fc3_a(x1)))
        x2 = self.activation(self.encoder_bn3_b(self.encoder_fc3_b(x2)))

        x1 = self.activation(self.encoder_bn4_a(self.encoder_fc4_a(x1)))
        x2 = self.activation(self.encoder_bn4_b(self.encoder_fc4_b(x2)))

        joint_x = torch.cat([x1, x2], 1)
        joint_x = self.activation(self.joint_encoder_bn5(self.joint_encoder_fc5(joint_x)))
        out = self.activation(self.joint_encoder_bn6(self.joint_encoder_fc6(joint_x)))
        out = self.joint_encoder_fc7(out)

        x1 = self.sigmoid(self.decoder_fc1_a(joint_x))
        x2 = self.sigmoid(self.decoder_fc1_b(joint_x))

        x1 = self.sigmoid(self.decoder_fc2_a(x1))
        x2 = self.sigmoid(self.decoder_fc2_b(x2))

        x1 = self.sigmoid(self.decoder_fc3_a(x1))
        x2 = self.sigmoid(self.decoder_fc3_b(x2))

        x1 = self.sigmoid(self.decoder_fc4_a(x1))
        x2 = self.sigmoid(self.decoder_fc4_b(x2))

        return (out, x1, x2, ori_x1, ori_x2)