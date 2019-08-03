import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    


# add residual block

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv1x2 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(1,1), stride=(1,1),padding=0,bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()


    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv1x2) #add 1x1 conv
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input):
        x = input
        # residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv1x2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # if (self.in_channels != self.out_channels):
        #     residual = self.conv1(residual)
        # x += residual
#         x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = F.avg_pool2d(x,kernel_size=2,stride=2)

        return x

    
class Vggish(nn.Module):
    def __init__(self, classes_num):
        
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock(in_channels=256, out_channels=512)

        self.fc_final = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)


    def forward(self, input, return_bottleneck=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        x1 = input.view(-1, 1, seq_len, mel_bins)
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

        x = F.log_softmax(self.fc_final(x), dim=-1)


        return x



if __name__ == '__main__':
    net = Vggish(10)
    print('net: {}'.format(net))
