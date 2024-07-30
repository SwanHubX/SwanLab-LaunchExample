import torch
import torch.nn as nn
import torch.optim as optim
import swanlab
import logging

# 0. 初始化实验环境
swanlab.init("Launch-Example", experiment_name="exp")  # 初始化swanlab
device = "cuda"
if not torch.cuda.is_available():  # 判断是否存在显卡，不存在则报WARNING
    device = "cpu"
    logging.warning("CUDA IS NOT AVAILIABLE, use cpu instead")

# 1. 定义公式，目标是一个简单的二元一次函数，定义域和值域都为[0,1]
func = lambda x: (2 * x - 1) ** 2

# 2. 定义模型，3层神经网络（增大参数或者层数效果会更好些）
model = nn.Sequential(nn.Linear(1, 16), nn.Sigmoid(), nn.Linear(16, 1))
model = model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 4. 训练模型
iters = 8000
batch_num = 256
for i in range(iters):
    # 生成数据
    x = torch.rand(batch_num, 1)
    y = func(x)
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()  # 梯度清零
    outputs = model(x)  # 前向传播
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if i % 5 == 0:
        print(f"Iter [{i+1}/{iters}], Loss: {loss.item():.4f}")
        swanlab.log({"loss": loss.item()}, step=i)

# 5. 验证模型
model.eval()
with torch.no_grad():
    sample_num = 101
    inputs = torch.linspace(0, 1, sample_num).unsqueeze(1).to(device)  # 从0到1线性采样
    outputs = func(inputs)
    predicts = model(inputs)
    # 绘制标注的图
    for i in range(sample_num):
        swanlab.log({"GroundTruth": outputs[i].item()}, step=i)
    # 绘制预测的图
    for i in range(sample_num):
        swanlab.log({"PredictFunc": predicts[i].item()}, step=i)
swanlab.finish()
