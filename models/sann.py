import torch
import torch.nn.functional as F
import torch.nn as nn


class SANN(nn.Module):
    def __init__(self, n_inp, n_out, t_inp, t_out, n_points, past_t, hidden_dim, dropout):
        super(SANN, self).__init__()
        # Variables
        self.n_inp = n_inp
        self.n_out = n_out
        self.t_inp = t_inp
        self.t_out = t_out
        self.n_points = n_points
        self.past_t = past_t
        self.hidden_dim = hidden_dim
        # Convolutional layer
        self.conv_block = AgnosticConvBlock(n_inp, n_points, past_t, hidden_dim,num_conv=1)
        self.convT = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,n_points))
        # Regressor layer
        self.regressor = ConvRegBlock(t_inp, t_out, n_points, hidden_dim)
        # Dropout
        self.drop = nn.Dropout2d(p=dropout)
        
    def forward(self, x):
        N,C,T,S = x.size()
        # Padding
        xp = F.pad(x, pad = (0,0,self.past_t-1,0))
        # NxCxTxS ---> NxHxTx1
        out = self.conv_block(xp)
        out = out.view(N,self.hidden_dim,T,1)
        # NxHxTx1 ---> NxHxTxS
        out = self.convT(out)
        # 2D dropout
        out = self.drop(out)
        # NxHxTxS ---> NxC'xT'xS
        out = self.regressor(out.view(N,-1,S))
        return out.view(N,self.n_out,self.t_out,self.n_points)
    
    
class AgnosticConvBlock(nn.Module):
    def __init__(self, n_inp, n_points, past_t, hidden_dim, num_conv):
        super(AgnosticConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=n_inp, out_channels=hidden_dim, kernel_size=(past_t, n_points), bias=True))
        layers.append(nn.BatchNorm2d(num_features=hidden_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)
    
    
    
class ConvRegBlock(nn.Module):
    def __init__(self, t_inp, t_out, n_points, hidden_dim):
        super(ConvRegBlock, self).__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels=hidden_dim*t_inp, out_channels=t_out, kernel_size=1, bias=True))
        layers.append(nn.BatchNorm1d(num_features=t_out, affine=True, track_running_stats=True))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)
    
    
        
