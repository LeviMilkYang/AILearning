import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
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