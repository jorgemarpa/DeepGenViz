import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        '''
        remove last "shompsize" elements in the last dimension
        '''
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        '''
        Temporal residual block containing:
            Dilated causal convolution
            WeightNorm
            ReLU
            Dropout
        n_inputs    : input size of residual block
        n_outputs   : ouput size of residual block
        kernel_size : conv kernel size
        stride      : conv stride
        dilation    : dilatation factor 2^i with i in [1,2,4,8,...]
        padding     : size of zero padding to the left

        for Conv1d, input cshape must be (batch_size, channels, seq_len)
        '''
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
        #                          self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2,
        #                          self.dropout2)
        # Followed by conditional 1x1 convolution to ensure element-wise
        # addition of block output and input are the same shape.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
                          if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                            self.conv2, self.chomp2, self.relu2, self.dropout2)
        out = net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalBlock_T(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super(TemporalBlock_T, self).__init__()
        '''
        Temporal residual block containing:
            Dilated causal convolution
            WeightNorm
            ReLU
            Dropout
        n_inputs    : input size of residual block
        n_outputs   : ouput size of residual block
        kernel_size : conv kernel size
        stride      : conv stride
        dilation    : dilatation factor 2^i with i in [1,2,4,8,...]
        padding     : size of zero padding to the left

        for Conv1d, input cshape must be (batch_size, channels, seq_len)
        '''
        self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs,
                                                    kernel_size,
                                                    stride=stride,
                                                    dilation=dilation,
                                                    padding=padding))
        self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs,
                                                    kernel_size,
                                                    stride=stride,
                                                    dilation=dilation,
                                                    padding=padding))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.ConvTranspose1d(n_inputs, n_outputs, 1) \
                          if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                            self.conv2, self.relu2, self.dropout2)
        out = net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2,
                 transpose=False):
        super(TemporalConvNet, self).__init__()
        '''
        num_inputs   : input dimension
        num_channels : [hidden_dim] * nlevels
                       lenght of num_channels is the number of levels
        '''
        layers = []
        num_levels = len(num_channels)
        if not transpose:
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                         stride=1, dilation=dilation_size,
                                         padding=(kernel_size-1) * dilation_size,
                                         dropout=dropout)]
        else:
            for i in reversed(range(num_levels)):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == num_levels-1 else num_channels[i-1]
                out_channels = num_channels[i]
                layers += [TemporalBlock_T(in_channels, out_channels,
                                           kernel_size,
                                           stride=1,
                                           dilation=dilation_size,
                                           padding=int((kernel_size-1) * dilation_size / 2),
                                           dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
