import numpy as np

# ==================== 1. 准备假数据 ====================
# 假设有 3 个样本，每个样本 2 个特征
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
# 真实的业务目标值（比如我们要拟合的房价或某个数值）
Y_true = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

# ==================== 2. 初始化参数 (Weights & Biases) ====================
# 现代推荐写法：不需要写 seed，它会自动从操作系统获取最高质量的随机种子
rng = np.random.default_rng()
# 方法 A：使用标准正态分布（有正有负），乘以小系数
W1 = rng.uniform(-0.5, 0.5, size=(2, 4))
b1 = np.zeros((4,))
W2 = rng.uniform(-0.5, 0.5, size=(4, 1))
b2 = np.zeros((1,))

learning_rate = 0.001 # 学习率（步长：每次调整参数不能迈太大步子）

# 模拟训练 1000 轮 (Epochs)
for epoch in range(30000):
    
    # ==================== 3. 前向传播 (Forward Pass) ====================
    Z1 = np.dot(X, W1) + b1          # 隐藏层线性变换 -> (3, 4)
    A1 = np.maximum(0, Z1)           # 隐藏层激活函数 (ReLU) -> (3, 4)
    Y_pred = np.dot(A1, W2) + b2     # 输出层预测值 -> (3, 1)
    
    # ==================== 4. 计算损失 (Loss) ====================
    error = Y_pred - Y_true
    loss = np.mean(error ** 2)       # 均方误差 (MSE)
    
    # ==================== 5. 反向传播 (Backward Pass / 矩阵求导与链式法则) ====================
    # N 是样本数量
    N = X.shape[0]
    
    # 5.1 输出层的梯度 (Loss 对预测值 Y_pred 的导数)
    # 对应公式 dL/dY_pred = 2 * (Y_pred - Y_true) / N
    dY_pred = (2 / N) * error        # 形状 (3, 1)
    
    # 5.2 第二层权重和偏置的梯度
    # 根据矩阵求导链式法则：dW2 = A1的转置 @ dY_pred
    dW2 = np.dot(A1.T, dY_pred)      # 形状 (4, 1)，完美对齐 W2 的形状！
    db2 = np.sum(dY_pred, axis=0)    # 形状 (1,)

    # 5.3 把误差往回传，通过第二层权重打到隐藏层 A1 上
    dA1 = np.dot(dY_pred, W2.T)      # 形状 (3, 4)

    # 5.4 穿透 ReLU 激活函数的导数
    # ReLU 的导数：当 Z1 > 0 时导数是 1，当 Z1 <= 0 时导数是 0
    dZ1 = dA1 * (Z1 > 0)             # 形状 (3, 4)
    
    # 5.5 第一层权重和偏置的梯度
    dW1 = np.dot(X.T, dZ1)           # 形状 (2, 4)，完美对齐 W1 的形状！
    db1 = np.sum(dZ1, axis=0)        # 形状 (4,)
    
    # ==================== 6. 梯度下降 (Gradient Descent / 更新参数) ====================
    # 让参数沿着梯度的反方向走一小步，从而减小未来的 Loss
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # 每隔 200 轮打印一次当前的 Loss，观察它是不是在乖乖下降
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("\n训练完成后的最终预测值：")
print(Y_pred)
print("真实的标签值：")
print(Y_true)