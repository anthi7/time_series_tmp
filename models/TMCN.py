import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        self.cnn_out_channel = 16
        
        self.CNN_Seasonal = CNN(configs, out_ch = self.cnn_out_channel)
        self.CNN_Trend = CNN(configs, out_ch = self.cnn_out_channel)
     
        self.Linear_Seasonal = nn.Linear(2*self.cnn_out_channel, 1)
        self.Linear_Trend = nn.Linear(2*self.cnn_out_channel, 1)

    def forward(self, x):
        #print("x:size = ", x.shape)
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # seasonal_init: [Batch, Channel, Input length]
        # trend_init: [Batch, Channel, Input length]

        seasonal_cnn = self.CNN_Seasonal(seasonal_init)
        trend_cnn = self.CNN_Trend(trend_init)
        
        seasonal_cnn, trend_cnn = seasonal_cnn.permute(0,2,1), trend_cnn.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_cnn)
        trend_output = self.Linear_Trend(trend_cnn)

        x = seasonal_output + trend_output

        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class CNN(nn.Module):
    """
    CNN
    """
    def __init__(self, configs, out_ch = 16):
        super(CNN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.channels = configs.enc_in
        tcn_layers = [4,4,0]

        self.cnn_out_channel = out_ch
        self.cnn_kernel_size_big = 16
        self.cnn_kernel_size_smol = 8
                
        self.in_embed = nn.Conv1d(configs.enc_in-1, self.cnn_out_channel, 1, padding='same')
        
        self.TCN_1_big = TemporalConvNet(self.cnn_out_channel, [self.cnn_out_channel]*tcn_layers[0]+[2*self.cnn_out_channel]*tcn_layers[1]+[4*self.cnn_out_channel]*tcn_layers[2])
        
        self.CNN_1_big = nn.Conv1d(self.cnn_out_channel, self.cnn_out_channel, self.cnn_kernel_size_big, padding='same')
        self.CNN_2_big =nn.Conv1d(self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size_big, padding='same')
        self.CNN_3_big =nn.Conv1d(2*self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size_big, padding='same')
        self.CNN_1_smol = nn.Conv1d(self.cnn_out_channel, self.cnn_out_channel, self.cnn_kernel_size_big, padding='same')
        self.CNN_2_smol =nn.Conv1d(self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size_smol, padding='same')
        self.CNN_3_smol =nn.Conv1d(2*self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size_smol, padding='same')
        
        self.Mix_1 = Mix(out_ch)
        self.Mix_2 = Mix(2*out_ch)
        self.Mix_3 = Mix(2*out_ch)
        
    def forward(self, x):
        #print("x:size = ", x.shape)
        # x: [Batch, Channel, Input length]
        x_emb = self.in_embed(x)
        
        x_tcn = self.TCN_1_big(x_emb)       
        
        x_cnn_b = self.CNN_1_big(x_emb)
        x_cnn_s = self.CNN_1_smol(x_emb)
        x_cnn_b, x_cnn_s = self.Mix_1(x_cnn_b, x_cnn_s)
        x_cnn_b, x_cnn_s = F.relu(x_cnn_b), F.relu(x_cnn_s)

        x_cnn_b = self.CNN_2_big(x_cnn_b)
        x_cnn_s = self.CNN_2_smol(x_cnn_s)
        x_cnn_b, x_cnn_s = self.Mix_2(x_cnn_b, x_cnn_s)
        x_cnn_b, x_cnn_s = F.relu(x_cnn_b), F.relu(x_cnn_s)

        x_cnn_b = self.CNN_3_big(x_cnn_b)
        x_cnn_s = self.CNN_3_smol(x_cnn_s)
        x_cnn_b, x_cnn_s = self.Mix_3(x_cnn_b, x_cnn_s)
        x_cnn_b, x_cnn_s = F.relu(x_cnn_b), F.relu(x_cnn_s)

        x_cnn = x_cnn_b + x_cnn_s

        return x_tcn + x_cnn 
    
class Mix(nn.Module):
    def __init__(self, out_ch = 16):
        super(Mix, self).__init__()
        self.Linear = nn.Linear(2*out_ch, 2*out_ch, bias=False)
        self.split_sizes = [out_ch, out_ch]

    def forward(self, x1, x2):
        x_cat = x1.permute(0,2,1), x2.permute(0,2,1)
        x_cat = torch.cat(x_cat, dim=-1)
        x_cat = x_cat + self.Linear(x_cat)
        x1, x2 = torch.split(x_cat, split_size_or_sections=self.split_sizes, dim=2)
        x1, x2 = x1.permute(0,2,1), x2.permute(0,2,1)
        return x1, x2