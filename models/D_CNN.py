import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    """
    Decomposition-CNN
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        self.cnn_out_channel = 16
        self.cnn_kernel_size = 8
        self.CNN_Seasonal = nn.Conv1d(configs.enc_in-1, self.cnn_out_channel, self.cnn_kernel_size, padding='same')
        self.CNN_Trend = nn.Conv1d(configs.enc_in-1, self.cnn_out_channel, self.cnn_kernel_size, padding='same')

        #self.cnn_out_size = self.seq_len - (self.cnn_kernel_size - 1)
            
        self.CNN_Seasonal_2 =nn.Conv1d(self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size, padding='same')
        self.CNN_Trend_2 =nn.Conv1d(self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size, padding='same')

        self.cnn_out_2_size = self.seq_len

        self.Linear_Seasonal = nn.Linear(2*self.cnn_out_channel, 1)
        self.Linear_Trend = nn.Linear(2*self.cnn_out_channel, 1)
        
        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        #print("x:size = ", x.shape)
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # seasonal_init: [Batch, Channel, Input length]
        # trend_init: [Batch, Channel, Input length]

        seasonal_cnn = F.relu(self.CNN_Seasonal(seasonal_init))
        seasonal_cnn = F.relu(self.CNN_Seasonal_2(seasonal_cnn))

        trend_cnn = F.relu(self.CNN_Trend(trend_init))
        trend_cnn = F.relu(self.CNN_Trend_2(trend_cnn))
        
        seasonal_cnn, trend_cnn = seasonal_cnn.permute(0,2,1), trend_cnn.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_cnn)
        trend_output = self.Linear_Trend(trend_cnn)

        x = seasonal_output + trend_output

        return x
