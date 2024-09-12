import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MELOSDP.SDP.model.conv import SDPClassifierCNN

def inference_conv(dataset_name='mc1', label_name='Defective'):
    # dataset_name = 'kc2'
    # label_name = 'problems'

    # 加载ARFF数据
    data, meta = arff.loadarff(f"data/{dataset_name}/test.arff")

    # 将ARFF数据转换为DataFrame
    df = pd.DataFrame(data)

    # 将目标向量转换为二进制
    # df["problems"] = df["problems"].apply(lambda x: 1 if x == b"yes" else 0)
    df[label_name] = df[label_name].apply(lambda x: 1 if x == b"Y" else 0)

    # 分离特征和标签
    x_test = df.drop(label_name, axis=1)
    y_test = df[label_name]

    # 数据标准化
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为张量
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # 加载模型
    model = SDPClassifierCNN().to(device)
    state_dict = torch.load(f"chkpt/{dataset_name}/conv/2/sdp_classifier_conv-300.pth", weights_only=True)
    model.load_state_dict(state_dict)

    # 测试
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor)
        # 将输出转换为二进制
        train_pred = outputs.argmax(dim=1).cpu().numpy()
        y_test_true = y_test_tensor.cpu().numpy()

        # 计算准确率
        test_accuracy = accuracy_score(y_test_true, train_pred)
        # 计算精确率
        test_precision = precision_score(y_test_true, train_pred)
        # 计算召回率
        test_recall = recall_score(y_test_true, train_pred)
        # 计算F1分数
        test_f1 = f1_score(y_test_true, train_pred)
        # 计算ROC AUC
        test_roc_auc = roc_auc_score(y_test_true, train_pred)

        print(
            f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}")

        return [round(metric, 2) for metric in [test_roc_auc, test_accuracy, test_f1, test_precision, test_recall]]


if __name__ == '__main__':
    result_list = inference_conv()
    print(result_list)
