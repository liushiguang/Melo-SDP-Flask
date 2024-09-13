from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 读取数据
def load_data(dataset_name='mc1', label_name='Defective'):
    # 加载数据并处理
    data, meta = arff.loadarff(f'SDP/data/{dataset_name}/train.arff')
    df = pd.DataFrame(data)

    # 预处理数据，将'Defective'列转化为二进制标签
    df[label_name] = df[label_name].apply(lambda x: 1 if x == b'Y' else 0)

    # 分离特征和目标变量
    X = df.drop(columns=[label_name]).values
    y = df[label_name].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return [X_train, X_test, y_train, y_test]

# 计算性能指标
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')

    return [round(metric, 2) for metric in [auc, accuracy, f1, precision, recall]]

# SVM（使用sklearn实现）
def supporting_vector_machine():
    X_train = load_data()[0]
    X_test = load_data()[1]
    y_train = load_data()[2]
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用RBF核的SVM，处理类别不平衡
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')  # 自动平衡类权重
    svm_model.fit(X_train, y_train)

    # 预测
    y_pred = svm_model.predict(X_test)

    return evaluate_performance(load_data()[3], y_pred)

# 随机森林
def random_forest(n_trees=100, max_depth=10):
    # 获取数据
    X_train = load_data()[0]
    X_test = load_data()[1]
    y_train = load_data()[2]

    rf_model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    return evaluate_performance(load_data()[3], y_pred)

# 朴素贝叶斯
def naive_bayes_classifier():
    # 获取数据
    X_train = load_data()[0]
    X_test = load_data()[1]
    y_train = load_data()[2]

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用高斯朴素贝叶斯
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # 预测
    y_pred = nb_model.predict(X_test)

    return evaluate_performance(load_data()[3], y_pred)


if __name__ == '__main__':
    # 使用支持向量机进行预测并评估
    svm_list = supporting_vector_machine()
    print("SVM预测结果:")
    print(svm_list)

    # 使用随机森林进行预测并评估
    rf_list = random_forest()
    print("随机森林预测结果:")
    print(rf_list)

    # 使用朴素贝叶斯进行预测并评估
    nbc_list = naive_bayes_classifier()
    print("朴素贝叶斯预测结果:")
    print(nbc_list)