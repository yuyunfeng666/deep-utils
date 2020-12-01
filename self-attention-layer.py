import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q,K,V计算与变形：

        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果没有mask，就生成一个

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：

        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘

        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x

