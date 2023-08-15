


#### positional encoding ####
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=16, num_layers=1, dropout=0.0, l_num=12):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(feature_size)  #位置编码前要做归一化，否则捕获不到位置信息
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2, dropout=dropout)  # 这里用了八个头
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)  # 这里用全连接层代替了decoder， 其实也可以加一下Transformer的decoder试一下效果
        self.init_weights()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(l_num, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = self.flatten(output)  # 扁平化 两维
        output = self.fc(output)
        return output



# model = TransAm(feature_size=10, num_layers=1, dropout=0.1, l_num=12)
# x = torch.randn(32, 12, 10)
# print(model(x).shape)  # torch.Size([32, 12, 1])

