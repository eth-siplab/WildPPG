import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=2, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(InceptionBlock, self).__init__()

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv2 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv3 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Adjust the convolutional layer when input and output channels differ
        if in_channels != out_channels:
            self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # Apply the residual connection if the input and output channels differ
        if residual.shape[1] != out.shape[1]:
            residual = self.conv_res(residual)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class UNET_1D_simp(nn.Module):
    def __init__(self, input_dim, output_dim, layer_n, kernel_size, depth, args):
        super(UNET_1D_simp, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_dim = output_dim
        self.args = args

        self.AvgPool1D0 = nn.AvgPool1d(kernel_size=int(self.args.fs/4), stride=None) if not args.data_type == 'ppg' else nn.AvgPool1d(kernel_size=int(args.fs/5), stride=1)
        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=2)
        self.AvgPoolOut = nn.AvgPool1d(kernel_size=6, stride=2, padding=2)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size, 1, 1)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n * 2), self.kernel_size, 2, 2)
        self.layer3 = self.down_layer(int(self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3),
                                      self.kernel_size, 2, 2)
        self.layer4 = self.down_layer(int(self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4),
                                      self.kernel_size, 2, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n * 7), int(self.layer_n * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n * 5), int(self.layer_n * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n * 3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1, padding=2)
        self.outcov2 = nn.Conv1d(in_channels=128, out_channels=181, kernel_size=1)
        self.fc = nn.Linear(output_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 181)
        self.out_act = nn.ReLU()

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        return nn.Sequential(*block)
    
    def forward(self, x): # x -> (batch_size, channels, time steps)
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x1 = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x1)

        x2 = torch.cat([out_2, pool_x2], 1)
        x3 = self.layer4(x2)

        #############Decoder####################
        up = self.upsample(x3)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)
        out1 = torch.tanh(out.squeeze())
        return out1, None
    
################## convnet ###################
class convnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, linear_unit, args):
        super(convnet, self).__init__()

        self.lin_unit = linear_unit
        self.args = args

        self.AvgPool1D0 = nn.AvgPool1d(kernel_size=int(self.args.fs/4), stride=None) if not args.data_type == 'ppg' else nn.AvgPool1d(kernel_size=int(args.fs/5), stride=1)

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size, stride, dilation=1)
        self.incept1 = InceptionBlock(out_channels, 12, kernel_size, stride, dilation=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept2 = InceptionBlock(36, 36, kernel_size, stride, dilation=1)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept3 = InceptionBlock(108, 108, kernel_size, stride, dilation=1)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=3)        
        self.conv_out = conbr_block(324, 324, 1, stride=1, dilation=1)
        self.fc1 = nn.Linear(3564, linear_unit)

    def forward(self, x):
        x = self.AvgPool1D0(x)
        x = self.conv1(x)
        x = self.incept1(x)
        x = self.pool1(x)
        x = self.incept2(x)
        x = self.pool2(x)
        x = self.incept3(x)
        x = self.pool3(x)      
        x = self.conv_out(x) 
        x = self.fc1(torch.flatten(x,start_dim=1))
        out = torch.tanh(x)
        return out.squeeze(), None
    
############################################### DCL Arch ############################

class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, data_type='ppg', conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units
        
        if data_type == 'ppg':
            self.fc1 = nn.Linear(128, 200)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = torch.tanh(x.squeeze())
        return x, None
    
############## Setup models #################

def setup_model(args, DEVICE):
    if args.model == 'unet':
        return UNET_1D_simp(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args).cuda(DEVICE)
    elif args.model == 'resunet':
        return resunet(args=args).cuda(DEVICE)
    elif args.model == 'convnet':
        return convnet(in_channels=1, out_channels=8, kernel_size=5, stride=1, linear_unit=args.out_dim, args=args).cuda(DEVICE)
    elif args.model == 'dcl':
        return DeepConvLSTM(n_channels=1, data_type=args.data_type, conv_kernels=64, kernel_size=5, LSTM_units=128).cuda(DEVICE)
    elif args.model == 'resnet1d':
        args.model == ResNet1D(in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, n_block=3, n_classes=args.out_dim, downsample_gap=2, increasefilter_gap=4, use_do=True).cuda(DEVICE)
    else:
        NotImplementedError

############ parameter count ###########
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############## RES-NET1D ################
"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, backbone=False, output_dim=200):
        super(ResNet1D, self).__init__()
        
        self.out_dim = output_dim
        self.backbone = backbone
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        self.dense2 = nn.Linear(out_channels, self.out_dim)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.transpose(-1,-2) # RESNET 1D takes channels first
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.backbone:
            out = self.dense2(out)
            return None, out
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_class = self.dense(out)
        if self.verbose:
            print('dense', out_class.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out_class.shape)
        
        return out_class, out    

        return q, attn       
