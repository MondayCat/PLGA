### GPU [显存不足时的Trick](https://blog.csdn.net/Zserendipity/article/details/105301983?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

此处不讨论多GPU， 分布式计算等情况

#### 降低batch size
这应该很好理解，适当降低batch size， 则模型每层的输入输出就会成线性减少， 效果相当明显。这里需要注意的一点是, dev batch size 的调整也有助于降低显存, 同时,不要将 dev 或 test 的batch size 设置为样本集长度, 我最近就干了这个傻事，害的我调试了一天才调出来是这个问题。
#### 选择更小的数据类型
一般默认情况下， 整个网络中采用的是32位的浮点数，如果切换到 16位的浮点数，其显存占用量将接近呈倍数递减。
#### 精简模型
在设计模型时，适当的精简模型，如原来两层的LSTM转为一层； 原来使用LSTM， 现在使用GRU； 减少卷积核数量； 尽量少的使用 Linear 等。
#### 数据角度
对于文本数据来说，长序列所带来的参数量是呈线性增加的， 适当的缩小序列长度可以极大的降低参数量。
#### total_loss
考虑到 loss 本身是一个包含梯度信息的 tensor， 因此，正确的求损失和的方式为：
total_loss += loss.item()
#### 释放不需要的张量和变量
采用del释放你不再需要的张量和变量，这也要求我们在写模型的时候注意变量的使用，不要随心所欲，漫天飞舞。
#### Relu 的 inplace 参数
激活函数 Relu() 有一个默认参数 inplace ，默认为Flase， 当设置为True的时候，我们在通过relu() 计算得到的新值不会占用新的空间而是直接覆盖原来的值，这表示设为True， 可以节省一部分显存。
#### 梯度累积
首先， 要了解一些Pytorch的基本知识：
> 在Pytorch 中，当我们执行 loss.backward() 时， 会为每个参数计算梯度，并将其存储在 paramter.grad 中， 注意到， paramter.grad 是一个张量， 其会累加每次计算得到的梯度。在 Pytorch 中， 只有调用 optimizer.step()时才会进行梯度下降更新网络参数。
我们知道， batch size 与占用显存息息相关，但有时候我们的batch size 又不能设置的太小，这咋办呢？ 答案就是梯度累加。
我们先来看看传统训练：
```
for i,(feature,target) in enumerate(train_loader):
    outputs = model(feature)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
 
    optimizer.zero_grad()   # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 反向传播， 更新网络参数
```
而加入梯度累加之后，代码是这样的：
```
for i,(features,target) in enumerate(train_loader):
    outputs = model(images)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
    loss = loss/accumulation_steps   # 可选，如果损失要在训练样本上取平均
 
    loss.backward()  # 计算梯度
    if((i+1)%accumulation_steps)==0:
        optimizer.step()        # 反向传播，更新网络参数
        optimizer.zero_grad()   # 清空梯度
```
比较来看， 我们发现，梯度累加本质上就是累加 accumulation_steps 个batch 的梯度， 再根据累加的梯度来更新网络参数，以达到类似batch_size 为 accumulation_steps * batch_size 的效果。在使用时，需要注意适当的扩大学习率。
更详细来说， 我们假设 batch size = 32， accumulation steps = 8 ， 梯度积累首先在前向传播的时候将 batch 分为 accumulation steps 份， 然后得到 size=4 的小份batch ， 每次就以小 batch 来计算梯度，但是不更新参数，将梯度积累下来，直到我们计算了 accumulation steps 个小 batch， 我们再更新参数。
梯度积累能很大程度上缓解GPU显存不足的问题，推荐使用。