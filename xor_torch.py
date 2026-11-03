import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备异或数据集 (转换为 PyTorch 的 Tensor)
X = torch.tensor([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.0, 1.0]], dtype=torch.float32)

Y_true = torch.tensor([[0.0], 
                       [1.0], 
                       [1.0], 
                       [0.0]], dtype=torch.float32)

# 2. 定义一个简单的两层神经网络 (MLP)
# nn.Sequential 就像搭积木一样把层串起来
model = nn.Sequential(
    nn.Linear(2, 4),  # 第一层：输入 2 维，输出 4 维 (相当于矩阵乘法 W1 @ X + b1)
    nn.ReLU(),        # 激活函数
    nn.Linear(4, 1),  # 第二层：输入 4 维，输出 1 维 (W2 @ A1 + b2)
    nn.Sigmoid()      # 输出层加个 Sigmoid 把结果压缩到 0~1 之间
)

# 3. 定义损失函数 (MSE) 和 优化器 (比如最经典的 Adam 或 SGD)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) # 传入模型参数，自动帮你更新

# 4. 训练循环
for epoch in range(20000):
    # 4.1 前向传播：计算预测值
    Y_pred = model(X)
    
    # 4.2 计算 Loss
    loss = criterion(Y_pred, Y_true)
    
    # 4.3 核心：自动求导三板斧
    optimizer.zero_grad()   # 1. 清空上一轮留下的历史梯度
    loss.backward()         # 2. 自动反向传播：PyTorch 在后台动态建图并算好所有导数！
    optimizer.step()        # 3. 梯度下降：自动更新权重
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("\nPyTorch 训练完成后的最终预测值：")
print(model(X).detach().numpy())