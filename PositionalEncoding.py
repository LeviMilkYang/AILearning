import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 创建一个足够长的主体矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算正弦和余弦的频率分母
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. 偶数维度填 sin，奇数维度填 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 注册成 buffer（告诉 PyTorch 这不是模型里需要随梯度更新的参数，但要随模型一起保存）
        self.register_buffer('pe', pe.unsqueeze(0)) # 形状变成 (1, max_len, d_model)

    def forward(self, x):
        # x 的形状是 (Batch, Seq_Len, d_model)
        seq_len = x.size(1)
        # 把对应长度的位置编码加到输入的词嵌入上 (E + P)
        return x + self.pe[:, :seq_len, :]