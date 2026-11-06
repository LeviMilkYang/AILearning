import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的特征维度 (例如 512 / 8 = 64)
        
        # 1. 定义 Q、K、V 的线性投影权重矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 2. 定义最后的线性融合矩阵 W_o (Output Projection)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 步骤 1：通过线性层生成 Q, K, V，并利用 reshape 和 transpose 拆分出多头
        # 形状变化: (B, N, d_model) -> (B, N, h, head_dim) -> 转置为 (B, h, N, head_dim)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 步骤 2 & 3：计算缩放点积得分 (Scaled Dot-Product)
        # Q @ K转置: (B, h, N, head_dim) @ (B, h, head_dim, N) -> 形状为 (B, h, N, N)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # 步骤 4：施加 Mask 掩码（若提供）
        if mask is not None:
            # 把 mask 中为 0（或 False）的位置填入负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 步骤 5：Softmax 归一化，得到权重矩阵
        attn_weights = F.softmax(scores, dim=-1)
        
        # 步骤 6：乘以 V 矩阵，得到每个头的加权特征
        # (B, h, N, N) @ (B, h, N, head_dim) -> 形状为 (B, h, N, head_dim)
        context = torch.matmul(attn_weights, V)
        
        # 步骤 7：Concat 拼接多头
        # 先把头和序列长度转置回来: (B, N, h, head_dim)，然后 reshape 成 (B, N, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 步骤 8：线性融合 (Linear Fusion / Output Projection)
        output = self.W_o(context)
        
        return output
# ==================== 验证运行的主程序 ====================
if __name__ == "__main__":
    # 1. 模拟输入：Batch Size = 2（两句话），Seq Len = 4（每句话4个词），d_model = 512（特征维度）
    x = torch.randn(2, 4, 512)
    
    # 2. 实例化我们刚才写的注意力模块（例如 8 个头）
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    
    # 3. 构造一个因果掩码 (Causal Mask)，形状为 (1, 1, 4, 4) 或 (4, 4)
    # 下三角为 1（允许看），上三角为 0（遮挡未来）
    mask = torch.tril(torch.ones(4, 4))
    
    # 4. 前向传播
    output = mha(x, mask=mask)
    
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
    print("运行成功！多头注意力模块工作正常。")