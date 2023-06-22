import random

# 设置随机种子
random.seed(2023)

# 读取原始数据
with open('guichu_danmaku_text.txt', 'r', encoding="utf-8") as f:
    data = f.read().splitlines()

# 随机打乱数据
random.shuffle(data)

# 计算数据集切分点
total_size = len(data)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

# 切分数据集
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# 将切分后的数据保存到不同的文件中
with open('train.txt', 'w', encoding="utf-8") as f:
    f.write('\n'.join(train_data))

with open('val.txt', 'w', encoding="utf-8") as f:
    f.write('\n'.join(val_data))

with open('test.txt', 'w', encoding="utf-8") as f:
    f.write('\n'.join(test_data))