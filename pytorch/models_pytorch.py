import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    

class BaselineCnn(nn.Module):
    def __init__(self, classes_num):
        
        super(BaselineCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), bias=False)

        self.fc1 = nn.Linear(512, classes_num, bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc1)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_bottleneck=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc1(x), dim=-1)

        return x
    

# add residual block

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)


    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=(1,1), stride=(1,1),padding=0,bias=False)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        # add 1x1 conv
        # self.conv = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=(1,1), stride=(1,1),padding=0)
        #
        # self.conv1 = nn.Conv2d(in_channels=out_channels,
        #                        out_channels=out_channels,
        #                        kernel_size=(1, 3), stride=(1, 1),
        #                        padding=(0, 1), bias=False)
        #
        # self.conv2 = nn.Conv2d(in_channels=out_channels,
        #                        out_channels=out_channels,
        #                        kernel_size=(3, 1), stride=(1, 1),
        #                        padding=(1, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        '''add cs attention'''
        # self.ca = ChannelAttention(out_channels)
        # self.sa = SpatialAttention(3)

        self.init_weights()


    def init_weights(self):
        # init_layer(self.conv) #add 1x1 conv
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input):
        x = input
        # residual = x
        # x = self.conv(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn2(self.conv2(x))
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = F.relu(x)
        # if (self.in_channels != self.out_channels):
        #     residual = self.conv1(residual)
        # x += residual
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x

    
class Vggish(nn.Module):
    def __init__(self, classes_num):
        
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock(in_channels=256, out_channels=512)


        self.fc_final = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)


    def forward(self, input, return_bottleneck=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # x1 = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # x1 = self.conv_block1(x1)
        # x1 = self.conv_block2(x1)
        # x1 = self.conv_block3(x1)
        # x1 = self.conv_block4(x1)
        #
        # x = [x, x1]

        # x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])
        # x = F.dropout(x)
        x = F.log_softmax(self.fc_final(x), dim=-1)


        return x



if __name__ == '__main__':
    net = Vggish(10)
    print('net: {}'.format(net))
    # input_img = torch.FloatTensor(1, 3, 128, 128)
    # input_img = Variable(input_img)
    # out = net(input_img)
