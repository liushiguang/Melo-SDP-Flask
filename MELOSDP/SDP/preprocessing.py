import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff
from utils import save_arff
import random

# dataset_name = 'kc2'
# dataset_name = 'mc1'
dataset_name = 'pc5'
# label_name = 'problems'
label_name = 'Defective'

# 读取ARFF文件
data, meta = arff.loadarff(f"dataset/{dataset_name.upper()}.arff")

# 将ARFF数据转换为DataFrame，并将字节类型转换为字符串
df = pd.DataFrame(data)
for column in df.select_dtypes([object]).columns:
    df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# 分离特征和标签
x = df.drop(label_name, axis=1)
y = df[label_name]

# 8:2分割数据集
random_seed = random.randint(0, 100)
print(f"Random Seed: {random_seed}")
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_seed)

# 训练集与测试集合并
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_val, y_val], axis=1)

# 将DataFrame转换为列表，准备写入ARFF
train_data_list = train_data.values.tolist()
test_data_list = test_data.values.tolist()

# 构造 attributes 列表
attributes = []
for name in df.columns:
    attr_type = meta[name][0]

    if attr_type == 'nominal':
        attr_type = list(meta[name][1])

    attributes.append((name, attr_type))

# 保存训练集为ARFF文件
save_arff(f'data/{dataset_name}/train.arff', dataset_name.upper(), attributes, train_data_list)

# 保存测试集为ARFF文件
save_arff(f'data/{dataset_name}/test.arff', dataset_name.upper(), attributes, test_data_list)

print("ARFF 文件已成功保存！")
