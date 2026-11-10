import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

# 1. 定义前馈网络 (FFN)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4  # 工业界惯例：通常是 d_model 的 4 倍
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # 使用平滑的 GeLU 激活函数
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# 2. 定义标准的 Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 第一个归一化：给多头注意力用
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        
        # 第二个归一化：给前馈网络用
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask=None):
        # 核心：Pre-LN 架构 (先归一化，再进计算，最后加上残差 x)
        # 1. 注意力子层
        x = x + self.attn(self.ln1(x), mask=mask)
        
        # 2. 前馈网络子层
        x = x + self.ffn(self.ln2(x))
        
        return x