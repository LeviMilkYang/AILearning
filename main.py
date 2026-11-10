import torch
import torch.nn as nn
import math
from SimpleTokenizer import SimpleTokenizer  # 假设你之前写的类保存在这个文件里
from TransformerBlock import TransformerBlock
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
    tf_block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    
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
    
    output = tf_block(x_with_pos, mask=mask)
    print("4. Transformer Block 输出的特征形状:", output.shape)
    print("--- 恭喜！输入到 Transformer Block 的整条流水线成功跑通！ ---")

    # 5. 加上语言模型头 (LM Head)，把隐藏特征映射回词表大小
    lm_head = nn.Linear(d_model, tokenizer.vocab_size)
    
    logits = lm_head(output)
    print("5. LM Head 输出的 Logits 形状:", logits.shape)
    
    # 6. 演示如何预测下一个字
    # 我们只看最后一个词预测出来的分数（logits 里取最后一个时间步）
    last_token_logits = logits[:, -1, :]  # 形状: (1, vocab_size)
    
    # 用 argmax 挑出概率（分数）最高的那个字 ID
    predicted_id = torch.argmax(last_token_logits, dim=-1).item()
    
    # 把 ID 解码成文字
    predicted_char = tokenizer.decode([predicted_id])
    print(f"输入文本: '{text}' -> 模型预测的下一个字符是: '{predicted_char}'")
    print("--- 恭喜！一个微型 GPT 的前向预测闭环彻底大功告成！ ---")