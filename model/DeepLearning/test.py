import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, C_size=17, d_model=128):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model,nhead=2, batch_first=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(C_size * d_model, 1)

    def forward(self, x):
        out = self.transformer(x, x)  # 输出为[N, 17, 128]
        out = self.flatten(out)  # 扁平化 两维
        out = self.fc(out)
        # out = torch.unsqueeze(out, dim=1)
        return out



model = Transformer(C_size=12, d_model=10)
data = torch.randn(32, 12, 10)
print(model(data).shape)
