import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from MELOSDP.SDP.model.conv import SDPClassifierCNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# dataset_name = 'kc2'
dataset_name = 'mc1'
# label_name = 'problems'
label_name = 'Defective'

# 读取ARFF文件
data, meta = arff.loadarff(f"data/{dataset_name}/train.arff")

# 将ARFF数据转换为DataFrame
df = pd.DataFrame(data)

# 将目标向量转换为二进制
# df["problems"] = df["problems"].apply(lambda x: 1 if x == b"yes" else 0)
df[label_name] = df[label_name].apply(lambda x: 1 if x == b"Y" else 0)

# 分离特征和标签
x = df.drop(label_name, axis=1)
y = df[label_name]

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 将数据转换为PyTorch的张量
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

def train_conv_model(seed):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型，损失函数和优化器
    model = SDPClassifierCNN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    writer = SummaryWriter(f"logs/{dataset_name}/conv/{seed}")
    # 训练模型
    epochs = 1000
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            train_loss = loss(outputs, y_batch)
            train_loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            # 创建模型保存目录
            os.makedirs(f"chkpt/{dataset_name}/conv/{seed}", exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), f"chkpt/{dataset_name}/conv/{seed}/sdp_classifier_conv-{epoch+1}.pth")

        if (epoch + 1) % 10 == 0:
            # 计算训练集上的指标和损失
            model.eval()
            with torch.no_grad():
                train_preds = []
                train_targets = []
                total_train_loss = 0.0
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    total_train_loss += loss(outputs, y_batch).item()
                    _, predicted = torch.max(outputs, 1)
                    train_preds.extend(predicted.cpu().numpy())
                    train_targets.extend(y_batch.cpu().numpy())

                train_accuracy = accuracy_score(train_targets, train_preds)
                train_precision = precision_score(train_targets, train_preds)
                train_recall = recall_score(train_targets, train_preds)
                train_auc = roc_auc_score(train_targets, train_preds)
                train_f1 = f1_score(train_targets, train_preds)
                avg_train_loss = total_train_loss / len(train_loader)


                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Precision/train", train_precision, epoch)
                writer.add_scalar("Recall/train", train_recall, epoch)
                writer.add_scalar("AUC/train", train_auc, epoch)
                writer.add_scalar("F1/train", train_f1, epoch)

                print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.4f}, "
                      f"Train Precision: {train_precision:.4f}, "
                      f"Train Recall: {train_recall:.4f}"
                      f"Train AUC: {train_auc:.4f}"
                      f"Train F1: {train_f1:.4f}")

            # 计算验证集上的指标和损失
            with torch.no_grad():
                test_preds = []
                test_targets = []
                total_test_loss = 0.0
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    total_test_loss += loss(outputs, y_batch).item()
                    _, predicted = torch.max(outputs, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_targets.extend(y_batch.cpu().numpy())

                test_accuracy = accuracy_score(test_targets, test_preds)
                test_precision = precision_score(test_targets, test_preds)
                test_recall = recall_score(test_targets, test_preds)
                train_auc = roc_auc_score(test_targets, test_preds)
                test_f1 = f1_score(test_targets, test_preds)
                avg_test_loss = total_test_loss / len(test_loader)

                writer.add_scalar("Loss/validation", avg_test_loss, epoch)
                writer.add_scalar("Accuracy/validation", test_accuracy, epoch)
                writer.add_scalar("Precision/validation", test_precision, epoch)
                writer.add_scalar("Recall/validation", test_recall, epoch)
                writer.add_scalar("AUC/validation", train_auc, epoch)
                writer.add_scalar("F1/validation", test_f1, epoch)

                print(f"Epoch {epoch + 1}: Validation Loss: {avg_test_loss:.4f}, "
                      f"Validation Accuracy: {test_accuracy:.4f}, "
                      f"Validation Precision: {test_precision:.4f}, "
                      f"Validation Recall: {test_recall:.4f}"
                      f"Validation AUC: {train_auc:.4f}"
                      f"Validation F1: {test_f1:.4f}")

for i in range(0, 20):
    train_conv_model(i)