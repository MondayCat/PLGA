### v1
scheduler控制new_data_flag作为判断客户端是否有收到新数据，启动训练过程的条件，模拟异步环境
```
new_data_flag = torch.rand(self.num_users)
activation_users = []
new_data_num = []
for index, val in enumerate(new_data_flag):
    if val < 0.5:
        activation_users.append(self.users[index])
        new_data_num.append(int(val*10))
```