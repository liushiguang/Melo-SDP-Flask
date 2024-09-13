import pandas as pd
from scipy.io import arff
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from MELOSDP.SDP.model.linear import SDPClassifierNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from preprocessing import dataset_name
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# dataset_name = 'kc2'
dataset_name = 'mc1'
# dataset_name = 'pc5'
# label_name = 'problems'
label_name = 'Defective'

# 读取ARFF文件
data, meta = arff.loadarff(f"data/{dataset_name}/train.arff")
df = pd.DataFrame(data)

# 将目标向量转换为二进制
# df["problems"] = df["problems"].apply(lambda x: 1 if x == b"yes" else 0)
df [label_name] = df[label_name].apply(lambda x: 1 if x == b"Y" else 0)

# 分离特征和标签
x = df.drop(label_name, axis=1).values
y = df[label_name].values

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

def train_linear_model(seed):
    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=seed)

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为张量
    x_train_tensor = torch.FloatTensor(x_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    x_val_tensor = torch.FloatTensor(x_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # 初始化模型
    input_size = x_train.shape[1]
    model = SDPClassifierNN(input_size).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    writer = SummaryWriter(f"logs/{dataset_name}/linear/{seed}")
    # 训练模型
    epochs = 1000
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor).squeeze()
        train_loss = loss(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            # 创建模型保存目录
            os.makedirs(f"chkpt/{dataset_name}/linear/{seed}", exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), f"chkpt/{dataset_name}/linear/{seed}/sdp_classifier_linear-{epoch+1}.pth")

        if (epoch + 1) % 10 == 0:
            # 将输出转换为二进制
            train_pred = outputs.ge(0.5).cpu().numpy()
            y_train_true = y_train_tensor.cpu().numpy()

            # 计算训练精确性和召回率
            train_accuracy = accuracy_score(y_train_true, train_pred)
            train_precision = precision_score(y_train_true, train_pred)
            train_recall = recall_score(y_train_true, train_pred)
            train_auc = roc_auc_score(y_train_true, train_pred)
            train_f1 = f1_score(y_train_true, train_pred)

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item()}')
            print(f'Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, AUC: {train_auc}, F1: {train_f1}')

            # 记录训练损失和准确性到 TensorBoard
            writer.add_scalar('Loss/train', train_loss.item(), epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Precision/train', train_precision, epoch)
            writer.add_scalar('Recall/train', train_recall, epoch)
            writer.add_scalar('AUC/train', train_auc, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)

            # 验证模型
            model.eval()
            with (torch.no_grad()):
                # 将输出转换为二进制
                outputs = model(x_val_tensor).squeeze()
                val_loss = loss(outputs, y_val_tensor)
                print(f"Validation Loss: {val_loss.item()}")

                y_pred = outputs.ge(0.5).cpu().numpy()
                y_val_true = y_val_tensor.cpu().numpy()

                # 计算验证精确性和召回率
                val_accuracy = accuracy_score(y_val_true, y_pred)
                val_precision = precision_score(y_val_true, y_pred)
                val_recall = recall_score(y_val_true, y_pred)
                val_auc = roc_auc_score(y_val_true, y_pred)
                val_f1 = f1_score(y_val_true, y_pred)

                print(f"Validation Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, AUC: {val_auc}, F1: {val_f1}")

                # 记录验证损失和准确性到 TensorBoard
                writer.add_scalar('Loss/validation', val_loss.item(), epoch)
                writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
                writer.add_scalar('Precision/validation', val_precision, epoch)
                writer.add_scalar('Recall/validation', val_recall, epoch)
                writer.add_scalar('AUC/validation', val_auc, epoch)
                writer.add_scalar('F1/validation', val_f1, epoch)

for i in range(0, 20):
    train_linear_model(i)