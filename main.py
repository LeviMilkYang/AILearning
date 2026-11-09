import torch
import torch.nn as nn
import math
from SimpleTokenizer import SimpleTokenizer  # 假设你之前写的类保存在这个文件里
from MultiHeadAttention import MultiHeadAttention
from PositionalEncoding import PositionalEncoding

# --- 主测试流程 ---
if __name__ == "__main__":
    # 模拟一段语料并初始化分词器
    corpus = "hello transformer, lets build a gpt step by step!"
    tokenizer = SimpleTokenizer(corpus)
    
    # 超参数设置
    d_model = 512
    num_heads = 8
    
    # 初始化我们今天的主角们
    embedding = nn.Embedding(tokenizer.vocab_size, d_model)
    pos_encoder = PositionalEncoding(d_model=d_model)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 模拟输入一句话
    text = "hello"
    input_ids = tokenizer.encode(text)  # 变成数字 ID 列表
    print("1. 文本转数字 ID:", input_ids)
    
    # 转成 PyTorch 张量，并升维加上 Batch 维度 -> 形状: (Batch=1, Seq_Len)
    x_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # 2. 词嵌入：查表变成稠密向量 -> 形状: (1, Seq_Len, 512)
    x_embed = embedding(x_ids)
    print("2. 词嵌入后的形状:", x_embed.shape)
    
    # 3. 注入位置编码：E + P -> 形状保持: (1, Seq_Len, 512)
    x_with_pos = pos_encoder(x_embed)
    print("3. 注入位置编码后的形状:", x_with_pos.shape)
    
    # 4. 送入你的多头注意力机制
    # 构造一个因果掩码 (根据当前的 Seq_Len 动态生成)
    seq_len = x_ids.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    output = mha(x_with_pos, mask=mask)
    print("4. 多头注意力输出的特征形状:", output.shape)
    print("--- 恭喜！输入到注意力的整条流水线成功跑通！ ---")